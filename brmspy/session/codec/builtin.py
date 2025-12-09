from __future__ import annotations

from typing import Any, Dict, List
import pickle

import numpy as np

from brmspy.helpers.log import log_warning
import xarray as xr
import arviz as az


from .base import Encoder, EncodeResult, ShmBlockSpec

ONE_MB = 1024 * 1024

class NumpyArrayCodec:
    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        arr = np.asarray(obj)
        data = arr.tobytes(order="C")
        nbytes = len(data)

        # Ask for exactly nbytes; OS may round up internally, that's fine.
        block = shm_pool.alloc(nbytes)
        block.shm.buf[:nbytes] = data

        meta: Dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "order": "C",
            "nbytes": nbytes,  # <-- critical
        }

        return EncodeResult(
            codec="numpy.ndarray",
            meta=meta,
            buffers=[ShmBlockSpec(name=block.name, size=block.size)],
        )

    def decode(self, meta: Dict[str, Any],
               buffers: List[memoryview]) -> Any:
        buf = buffers[0]
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        nbytes = int(meta["nbytes"])

        # Only use the slice that actually holds array data
        view = buf[:nbytes]
        arr = np.frombuffer(view, dtype=dtype)
        arr = arr.reshape(shape)
        return arr

class PickleCodec:
    def can_encode(self, obj: Any) -> bool:
        # Fallback – always True
        return True

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        block = shm_pool.alloc(len(data))
        block.shm.buf[:len(data)] = data

        meta: Dict[str, Any] = {"length": len(data)}

        size_bytes = len(data)
        if size_bytes > ONE_MB:
            size_mb = size_bytes / ONE_MB
            log_warning(
                f"PickleCodec encoding large object: type={type(obj)}, size={size_mb:,.2f} MB"
            )

        return EncodeResult(
            codec="pickle",
            meta=meta,
            buffers=[ShmBlockSpec(name=block.name, size=block.size)],
        )

    def decode(self, meta: Dict[str, Any],
               buffers: List[memoryview]) -> Any:
        buf = buffers[0]
        length = meta["length"]
        payload = bytes(buf[:length])
        return pickle.loads(payload)


class InferenceDataCodec(Encoder):
    """Encode arviz.InferenceData by pushing its underlying arrays into shm."""

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, az.InferenceData)

    def encode(self, obj: az.InferenceData, shm_pool: Any) -> EncodeResult:
        buffers: List[ShmBlockSpec] = []
        groups_meta: Dict[str, Any] = {}
        total_bytes = 0

        # Walk each group: posterior, posterior_predictive, etc.
        for group_name in obj.groups():
            ds: xr.Dataset = getattr(obj, group_name)
            g_meta: Dict[str, Any] = {
                "data_vars": {},
                "coords": {},
            }

            # COORDS: generally smaller, but can be arrays.
            for cname, coord in ds.coords.items():
                values = np.asarray(coord.values)
                if values.dtype.kind in "iufb":  # numeric-ish
                    data = values.tobytes(order="C")
                    nbytes = len(data)
                    block = shm_pool.alloc(nbytes)
                    block.shm.buf[:nbytes] = data

                    buffer_idx = len(buffers)
                    buffers.append(
                        ShmBlockSpec(name=block.name, size=block.size)
                    )
                    total_bytes += nbytes

                    g_meta["coords"][cname] = {
                        "kind": "array",
                        "buffer_idx": buffer_idx,
                        "dtype": str(values.dtype),
                        "shape": list(values.shape),
                        "dims": list(coord.dims),
                        "nbytes": nbytes,
                    }
                else:
                    # Non-numeric / object coords: keep them small & pickle in meta.
                    g_meta["coords"][cname] = {
                        "kind": "pickle",
                        "dims": list(coord.dims),
                        "payload": pickle.dumps(coord.values, protocol=pickle.HIGHEST_PROTOCOL),
                    }

            # DATA VARS: main heavy arrays
            for vname, da in ds.data_vars.items():
                arr = np.asarray(da.data)
                data = arr.tobytes(order="C")
                nbytes = len(data)

                block = shm_pool.alloc(nbytes)
                block.shm.buf[:nbytes] = data

                buffer_idx = len(buffers)
                buffers.append(
                    ShmBlockSpec(name=block.name, size=block.size)
                )
                total_bytes += nbytes

                g_meta["data_vars"][vname] = {
                    "buffer_idx": buffer_idx,
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "dims": list(da.dims),
                    "nbytes": nbytes,
                }

            groups_meta[group_name] = g_meta

        if total_bytes > ONE_MB:
            size_mb = total_bytes / ONE_MB
            log_warning(
                f"InferenceDataCodec encoding large InferenceData: "
                f"size={size_mb:,.2f} MB, groups={list(groups_meta.keys())}"
            )

        meta: Dict[str, Any] = {
            "groups": groups_meta,
            "codec_version": 1,
        }

        return EncodeResult(
            codec="arviz.InferenceData",
            meta=meta,
            buffers=buffers,
        )

    def decode(self, meta: Dict[str, Any],
               buffers: List[memoryview]) -> Any:
        groups_meta = meta["groups"]
        groups: Dict[str, xr.Dataset] = {}

        for group_name, g_meta in groups_meta.items():
            data_vars = {}
            coords = {}

            # Rebuild coords
            for cname, cmeta in g_meta["coords"].items():
                kind = cmeta["kind"]
                if kind == "array":
                    buf = buffers[cmeta["buffer_idx"]]
                    nbytes = int(cmeta["nbytes"])
                    view = buf[:nbytes]
                    arr = np.frombuffer(
                        view,
                        dtype=np.dtype(cmeta["dtype"])
                    ).reshape(cmeta["shape"])
                    coords[cname] = (tuple(cmeta["dims"]), arr)
                elif kind == "pickle":
                    values = pickle.loads(cmeta["payload"])
                    coords[cname] = (tuple(cmeta["dims"]), values)
                else:
                    raise ValueError(f"Unknown coord kind: {kind!r}")

            # Rebuild data_vars
            for vname, vmeta in g_meta["data_vars"].items():
                buf = buffers[vmeta["buffer_idx"]]
                nbytes = int(vmeta["nbytes"])
                view = buf[:nbytes]
                arr = np.frombuffer(
                    view,
                    dtype=np.dtype(vmeta["dtype"])
                ).reshape(vmeta["shape"])
                data_vars[vname] = (tuple(vmeta["dims"]), arr)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
            )
            groups[group_name] = ds

        # Construct InferenceData from datasets
        idata = az.InferenceData(**groups, warn_on_custom_groups=False)
        return idata