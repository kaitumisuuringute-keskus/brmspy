from __future__ import annotations

"""
Built-in IPC codecs used by the session layer (internal).

These codecs serialize values that cross the main↔worker boundary:

- Large numeric payloads are stored in shared memory (SHM) and only `(name, size)`
  references plus compact metadata are sent over the `Pipe`.
- Small/irregular payloads fall back to pickling (still stored in SHM to avoid
  pipe size limits).

All codecs follow the `Encoder` protocol from [`brmspy.types.session`][brmspy.types.session].
"""

import pickle
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from brmspy.helpers.log import log_warning
from brmspy._session.codec.base import CodecRegistry
from brmspy.types.session import EncodeResult, Encoder

from ...types.shm_extensions import ShmArray, ShmDataFrameColumns, ShmDataFrameSimple
from ...types.shm import ShmBlock, ShmBlockSpec

ONE_MB = 1024 * 1024


def array_order(a: np.ndarray) -> Literal["C", "F", "non-contiguous"]:
    """
    Determine how an array can be reconstructed from a raw buffer.

    Returns `"C"` for C-contiguous arrays, `"F"` for Fortran-contiguous arrays,
    otherwise `"non-contiguous"` (meaning: bytes were obtained by forcing
    a contiguous copy during encoding).
    """
    if a.flags["C_CONTIGUOUS"]:
        return "C"
    if a.flags["F_CONTIGUOUS"]:
        return "F"
    return "non-contiguous"


class NumpyArrayCodec:
    """SHM-backed codec for [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)."""

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        if isinstance(obj, ShmArray):
            arr = obj
            nbytes = obj.block.size
            block = obj.block
        else:
            log_warning("np.ndarray not in SHM, storing!")
            arr = np.asarray(obj)
            data = arr.tobytes(order="C")
            nbytes = len(data)

            # Ask for exactly nbytes; OS may round up internally, that's fine.
            block = shm_pool.alloc(nbytes)
            block.shm.buf[:nbytes] = data

        meta: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "order": array_order(arr),
            "nbytes": nbytes,  # <-- critical
        }

        return EncodeResult(
            codec=type(self).__name__,
            meta=meta,
            buffers=[ShmBlockSpec(name=block.name, size=block.size)],
        )

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        buf = buffers[0]
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        nbytes = int(meta["nbytes"])
        order = meta["order"]
        assert buf.shm.buf
        memview = memoryview(buf.shm.buf)

        # Only use the slice that actually holds array data
        view = memview[:nbytes]
        arr = np.ndarray(shape=shape, dtype=dtype, buffer=view, order=order)
        return arr


class PandasDFCodec:
    """
    SHM-backed codec for numeric-only [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

    Object-dtype columns are intentionally rejected to avoid surprising implicit
    conversions; those cases fall back to pickle.
    """

    def can_encode(self, obj: Any) -> bool:
        if not isinstance(obj, pd.DataFrame):
            return False

        if any(obj[c].dtype == "O" for c in obj.columns):
            log_warning(
                "pd.DataFrame contains Object type columns, falling back to pickle!"
            )
            return False
        return True

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        assert isinstance(obj, pd.DataFrame)  # assert type

        meta: dict[str, Any] = {
            "columns": list(obj.columns),
            "index": list(obj.index),
            "variant": "single",
        }
        buffers: list[ShmBlockSpec] = []

        if obj.empty:
            meta["variant"] = "empty"
        elif isinstance(obj, ShmDataFrameSimple):
            # single dtype matrix
            meta["variant"] = "single"
            meta["dtype"] = str(obj.values.dtype)
            meta["order"] = array_order(obj.values)
            spec = ShmBlockSpec(name=obj.block.name, size=obj.block.size)
            buffers.append(spec)
        elif isinstance(obj, ShmDataFrameColumns):
            # per column buffers
            meta["variant"] = "columnar"
            meta["order"] = "F"
            for column in obj.columns:
                block = obj.blocks_columns[column]
                spec = ShmBlockSpec(name=block.name, size=block.size)
                buffers.append(spec)
        else:
            # Fallback: put each column in its own SHM block
            meta["variant"] = "columnar"
            meta["order"] = "C"
            dtypes: list[str] = []

            for col_name in obj.columns:
                col = obj[col_name]

                # For now, don't silently encode object-dtype columns
                if col.dtype == "O":
                    raise TypeError(
                        f"Cannot SHM-encode object-dtype column {col_name!r}; some values: {col.unique()[:10]} "
                        "convert to numeric/categorical or add a dedicated object codec."
                    )

                values = np.asarray(col.to_numpy(copy=False), order="C")
                dtypes.append(str(values.dtype))

                data = values.tobytes(order="C")
                nbytes = len(data)

                block = shm_pool.alloc(nbytes)
                block.shm.buf[:nbytes] = data

                spec = ShmBlockSpec(name=block.name, size=nbytes)
                buffers.append(spec)

            meta["dtypes"] = dtypes

        return EncodeResult(codec=type(self).__name__, meta=meta, buffers=buffers)

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        if meta.get("variant") == "empty":
            return pd.DataFrame({})
        elif meta.get("variant") == "single":
            assert buffers[0].shm.buf
            memview = memoryview(buffers[0].shm.buf)
            buf = memview.cast("B")
            spec = buffer_specs[0]
            dtype = np.dtype(meta["dtype"])
            nbytes = spec["size"]
            order = meta["order"]

            columns = meta["columns"]
            index = meta["index"]
            shape = (len(index), len(columns))

            # Only use the slice that actually holds array data
            view = buf[:nbytes]
            arr = np.ndarray(shape=shape, dtype=dtype, buffer=view, order=order)

            df = ShmDataFrameSimple(data=arr, index=index, columns=columns)
            df.block = ShmBlockSpec(name=spec["name"], size=spec["size"])

            return df
        elif meta.get("variant") == "columnar":
            columns = meta["columns"]
            index = meta["index"]
            dtypes = meta["dtypes"]
            nrows = len(index)

            if len(columns) != len(buffers) or len(columns) != len(dtypes):
                raise ValueError(
                    f"Columnar decode mismatch: "
                    f"{len(columns)} columns, {len(buffers)} buffers, {len(dtypes)} dtypes"
                )

            data: dict[str, np.ndarray] = {}

            for i, col_name in enumerate(columns):
                dtype = np.dtype(dtypes[i])
                spec = buffer_specs[i]
                buf = buffers[i].shm.buf
                assert buf
                memview = memoryview(buf)
                buf = memview.cast("B")
                nbytes = spec["size"]

                # 1D column
                view = buf[:nbytes]
                arr = np.ndarray(shape=(nrows,), dtype=dtype, buffer=view, order="C")
                data[col_name] = arr

            # You can swap this to ShmDataFrameColumns if you want that type
            df = pd.DataFrame(data=data, index=index)
            return df
        else:
            raise Exception(f"Unknown DataFrame variant {meta.get('variant')}")


class PickleCodec:
    """
    Pickle fallback codec (internal).

    Always encodes successfully, so it must be registered last. The pickled bytes
    are still stored in SHM to keep pipe traffic small and bounded.
    """

    def can_encode(self, obj: Any) -> bool:
        # Fallback – always True
        return True

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        block = shm_pool.alloc(len(data))
        block.shm.buf[: len(data)] = data

        meta: dict[str, Any] = {"length": len(data)}

        size_bytes = len(data)
        if size_bytes > ONE_MB:
            size_mb = size_bytes / ONE_MB
            log_warning(
                f"PickleCodec encoding large object: type={type(obj)}, size={size_mb:,.2f} MB"
            )

        return EncodeResult(
            codec=type(self).__name__,
            meta=meta,
            buffers=[ShmBlockSpec(name=block.name, size=block.size)],
        )

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        block = buffers[0]
        assert block.shm.buf
        buf = memoryview(block.shm.buf)
        length = meta["length"]
        payload = bytes(buf[:length])
        return pickle.loads(payload)


class InferenceDataCodec(Encoder):
    """Encode arviz.InferenceData by pushing its underlying arrays into shm."""

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, az.InferenceData)

    def encode(self, obj: az.InferenceData, shm_pool: Any) -> EncodeResult:
        buffers: list[ShmBlockSpec] = []
        groups_meta: dict[str, Any] = {}
        total_bytes = 0

        # Walk each group: posterior, posterior_predictive, etc.
        for group_name in obj.groups():
            ds: xr.Dataset = getattr(obj, group_name)
            g_meta: dict[str, Any] = {
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
                    buffers.append(ShmBlockSpec(name=block.name, size=block.size))
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
                        "payload": pickle.dumps(
                            coord.values, protocol=pickle.HIGHEST_PROTOCOL
                        ),
                    }

            # DATA VARS: main heavy arrays
            for vname, da in ds.data_vars.items():
                arr = np.asarray(da.data)
                data = arr.tobytes(order="C")
                nbytes = len(data)

                block = shm_pool.alloc(nbytes)
                block.shm.buf[:nbytes] = data

                buffer_idx = len(buffers)
                buffers.append(ShmBlockSpec(name=block.name, size=block.size))
                total_bytes += nbytes

                g_meta["data_vars"][vname] = {
                    "buffer_idx": buffer_idx,
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "dims": list(da.dims),
                    "nbytes": nbytes,
                }

            groups_meta[group_name] = g_meta

        meta: dict[str, Any] = {
            "groups": groups_meta,
            "codec_version": 1,
        }

        return EncodeResult(
            codec=type(self).__name__,
            meta=meta,
            buffers=buffers,
        )

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        groups_meta = meta["groups"]
        groups: dict[str, xr.Dataset] = {}

        for group_name, g_meta in groups_meta.items():
            data_vars = {}
            coords = {}

            # Rebuild coords
            for cname, cmeta in g_meta["coords"].items():
                kind = cmeta["kind"]
                if kind == "array":
                    block = buffers[cmeta["buffer_idx"]]
                    assert block.shm.buf
                    buf = memoryview(block.shm.buf)
                    nbytes = int(cmeta["nbytes"])
                    view = buf[:nbytes]
                    arr = np.frombuffer(view, dtype=np.dtype(cmeta["dtype"])).reshape(
                        cmeta["shape"]
                    )
                    coords[cname] = (tuple(cmeta["dims"]), arr)
                elif kind == "pickle":
                    values = pickle.loads(cmeta["payload"])
                    coords[cname] = (tuple(cmeta["dims"]), values)
                else:
                    raise ValueError(f"Unknown coord kind: {kind!r}")

            # Rebuild data_vars
            for vname, vmeta in g_meta["data_vars"].items():
                block = buffers[vmeta["buffer_idx"]]
                assert block.shm.buf
                buf = memoryview(block.shm.buf)
                nbytes = int(vmeta["nbytes"])
                view = buf[:nbytes]
                arr = np.frombuffer(view, dtype=np.dtype(vmeta["dtype"])).reshape(
                    vmeta["shape"]
                )
                data_vars[vname] = (tuple(vmeta["dims"]), arr)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
            )
            groups[group_name] = ds

        # Construct InferenceData from datasets
        idata = az.InferenceData(**groups, warn_on_custom_groups=False)
        return idata


class GenericDataClassCodec(Encoder):
    """
    Generic codec for dataclasses (internal).

    Encodes each `init=True` field by delegating to a
    [`CodecRegistry`][brmspy._session.codec.base.CodecRegistry]. Use `skip_fields` to exclude
    fields that must not cross the boundary.
    """

    def __init__(
        self,
        cls: type[Any],
        registry: CodecRegistry,
        *,
        skip_fields: set[str] | None = None,
    ) -> None:
        if not is_dataclass(cls):
            raise TypeError(f"{cls!r} is not a dataclass")

        self._cls = cls
        self._registry = registry
        self.codec = f"dataclass::{cls.__module__}.{cls.__qualname__}"

        self._skip_fields = skip_fields or set()
        self._field_names: list[str] = []

        # Precompute which fields we actually encode
        for f in dc_fields(cls):
            if not f.init:
                continue
            if f.name in self._skip_fields:
                continue
            self._field_names.append(f.name)

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, self._cls)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        buffers: list[ShmBlockSpec] = []
        fields_meta: dict[str, Any] = {}

        for field_name in self._field_names:
            value = getattr(obj, field_name)

            # Delegate to registry; chooses right encoder for the actual *runtime* type
            res = self._registry.encode(value, shm_pool)

            start = len(buffers)
            count = len(res.buffers)

            fields_meta[field_name] = {
                "codec": res.codec,
                "meta": res.meta,
                "start": start,
                "count": count,
            }

            buffers.extend(res.buffers)

        meta: dict[str, Any] = {
            "module": self._cls.__module__,
            "qualname": self._cls.__qualname__,
            "fields": fields_meta,
        }

        return EncodeResult(
            codec=self.codec,
            meta=meta,
            buffers=buffers,
        )

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any:
        fields_meta: dict[str, Any] = meta["fields"]
        kwargs: dict[str, Any] = {}

        for field_name, fmeta in fields_meta.items():
            codec_name = fmeta["codec"]
            start = fmeta["start"]
            count = fmeta["count"]

            # IMPORTANT: slice buffer_specs in the same way as buffers
            value = self._registry.decode(
                codec_name,
                fmeta["meta"],
                buffers[start : start + count],
                buffer_specs[start : start + count],
                shm_pool,
            )
            kwargs[field_name] = value

        return self._cls(**kwargs)
