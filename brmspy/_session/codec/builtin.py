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

from collections.abc import Callable
import pickle
from dataclasses import dataclass, fields as dc_fields
from dataclasses import is_dataclass
from typing import Any, Literal, TypedDict

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from brmspy.helpers.log import log_warning
from brmspy._session.codec.base import CodecRegistry
from brmspy.types.session import EncodeResult, Encoder, PayloadRef

from ...types.shm_extensions import (
    PandasColumnMetadata,
    ShmArray,
    ShmDataFrameColumns,
    ShmDataFrameSimple,
)
from ...types.shm import ShmBlock, ShmRef

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


class NumpyArrayCodec(Encoder):
    """SHM-backed codec for [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)."""

    @classmethod
    def is_string_object(cls, a: np.ndarray, sample: int = 1000):
        if a.dtype != object:
            return False
        it = a.flat
        for _ in range(min(sample, a.size)):
            v = next(it, None)
            if v is not None and not isinstance(v, str):
                return False
        return True

    @classmethod
    def to_shm(cls, obj: np.ndarray, shm_pool: Any) -> EncodeResult:
        arr = np.asarray(obj)
        is_object = arr.dtype == "O"
        is_string = cls.is_string_object(obj)

        if isinstance(arr, ShmArray):
            arr = arr
            nbytes = arr.block["content_size"]
            ref = arr.block

        else:
            if not is_object:
                data = arr.tobytes(order="C")
            elif is_string:
                arr = arr.astype("U")
                data = arr.tobytes(order="C")
            else:
                data = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)

            nbytes = len(data)

            # Ask for exactly nbytes; OS may round up internally, that's fine.
            block = shm_pool.alloc(nbytes)
            block.shm.buf[:nbytes] = data
            ref = ShmRef(
                name=block.name, size=block.size, content_size=block.content_size
            )

        meta: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "order": array_order(arr),
        }

        return EncodeResult(
            codec=cls.__name__,
            meta=meta,
            buffers=[ref],
        )

    @classmethod
    def from_shm(cls, ref: PayloadRef, block: ShmBlock) -> np.ndarray:
        meta = ref["meta"]
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        order = meta["order"]

        return ShmArray.from_block(block=block, shape=shape, dtype=dtype, order=order)

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        return NumpyArrayCodec.to_shm(obj, shm_pool)

    def decode(
        self,
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        buf, _ = get_buf(payload["buffers"][0])

        return NumpyArrayCodec.from_shm(payload, buf)


class PandasDFCodec(Encoder):
    """
    SHM-backed codec for numeric-only [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

    Object-dtype columns are intentionally rejected to avoid surprising implicit
    conversions; those cases fall back to pickle.
    """

    def can_encode(self, obj: Any) -> bool:
        if not isinstance(obj, pd.DataFrame):
            return False

        return True

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        assert isinstance(obj, pd.DataFrame)  # assert type

        meta: dict[str, Any] = {
            "columns": list(obj.columns),
            "index": list(obj.index),
            "variant": "single",
        }
        buffers: list[ShmRef] = []

        if obj.empty:
            meta["variant"] = "empty"
        elif isinstance(obj, ShmDataFrameSimple):
            # single dtype matrix
            meta["variant"] = "single"
            meta["dtype"] = str(obj.values.dtype)
            meta["order"] = array_order(obj.values)
            buffers.append(obj.block)
        elif isinstance(obj, ShmDataFrameColumns):
            # per column buffers
            meta["variant"] = "columnar"
            meta["order"] = "F"
            meta["columns"] = obj.blocks_columns
        else:
            # Fallback: put each column in its own SHM block
            meta["variant"] = "columnar"
            meta["order"] = "C"
            columns: dict[str, PandasColumnMetadata] = {}

            for col_name in obj.columns:
                col = obj[col_name]

                encoded = NumpyArrayCodec.to_shm(col.to_numpy(copy=False), shm_pool)
                block = encoded.buffers[0]

                spec = ShmRef(
                    name=block["name"],
                    size=block["size"],
                    content_size=block["content_size"],
                )
                columns[col_name] = {
                    "name": col_name,
                    "block": spec,
                    "np_dtype": encoded.meta["dtype"],
                    "pd_type": str(col.dtype),
                    "params": {},
                }
            meta["columns"] = columns

        return EncodeResult(codec=type(self).__name__, meta=meta, buffers=buffers)

    def decode(
        self,
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        meta = payload["meta"]
        buffer_specs = payload["buffers"]

        if meta.get("variant") == "empty":
            return pd.DataFrame({})

        if meta.get("variant") == "single":
            spec = buffer_specs[0]
            buf, memview = get_buf(buffer_specs[0])
            dtype = np.dtype(meta["dtype"])
            nbytes = spec["size"]
            order = meta["order"]

            columns = meta["columns"]
            index = meta["index"]
            shape = (len(index), len(columns))

            # Only use the slice that actually holds array data
            view = memview[:nbytes]
            arr = np.ndarray(shape=shape, dtype=dtype, buffer=view, order=order)

            df = ShmDataFrameSimple(data=arr, index=index, columns=columns)
            df.block = spec

            return df
        elif meta.get("variant") == "columnar":
            columns_metadata: dict[str, PandasColumnMetadata] = meta["columns"]
            index = meta["index"]
            nrows = len(index)

            columns = list(columns_metadata.keys())

            data: dict[str, np.ndarray] = {}

            for i, col_name in enumerate(columns):
                col_name = str(col_name)
                metadata = columns_metadata[col_name]
                dtype = np.dtype(metadata["np_dtype"])
                spec = metadata["block"]
                buf, view = get_buf(spec)
                nbytes = spec["content_size"]

                # 1D column
                view = view[:nbytes]
                arr = ShmArray.from_block(
                    block=buf, shape=(nrows,), dtype=dtype, order="C"
                )
                data[col_name] = arr

            df = ShmDataFrameColumns(data=data, index=index)
            df.blocks_columns = columns_metadata
            return df
        else:
            raise Exception(f"Unknown DataFrame variant {meta.get('variant')}")


class PickleCodec(Encoder):
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
        if size_bytes > ONE_MB * 10:
            size_mb = size_bytes / ONE_MB
            log_warning(
                f"PickleCodec encoding large object: type={type(obj)}, size={size_mb:,.2f} MB"
            )

        return EncodeResult(
            codec=type(self).__name__,
            meta=meta,
            buffers=[
                ShmRef(
                    name=block.name, size=block.size, content_size=block.content_size
                )
            ],
        )

    def decode(
        self,
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        specs = payload["buffers"]
        block, buf = get_buf(specs[0])
        length = block.content_size
        b = bytes(buf[:length])
        return pickle.loads(b)


class InferenceDataCodec(Encoder):
    """Encode arviz.InferenceData by pushing its underlying arrays into shm."""

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, az.InferenceData)

    def encode(self, obj: az.InferenceData, shm_pool: Any) -> EncodeResult:
        buffers: list[ShmRef] = []
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
                    buffers.append(
                        ShmRef(
                            name=block.name,
                            size=block.size,
                            content_size=block.content_size,
                        )
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
                        "payload": pickle.dumps(
                            coord.values, protocol=pickle.HIGHEST_PROTOCOL
                        ),
                    }

            # DATA VARS: main heavy arrays
            for vname, da in ds.data_vars.items():
                arr = np.asarray(da.data)
                encoded = NumpyArrayCodec.to_shm(arr, shm_pool)
                meta = encoded.meta
                spec = encoded.buffers[0]
                nbytes = spec["content_size"]

                buffer_idx = len(buffers)
                buffers.append(spec)
                total_bytes += nbytes

                g_meta["data_vars"][vname] = {
                    "buffer_idx": buffer_idx,
                    "dtype": str(meta["dtype"]),
                    "shape": list(meta["shape"]),
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
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        meta = payload["meta"]
        specs = payload["buffers"]
        groups_meta = meta["groups"]
        groups: dict[str, xr.Dataset] = {}

        for group_name, g_meta in groups_meta.items():
            data_vars = {}
            coords = {}

            # Rebuild coords
            for cname, cmeta in g_meta["coords"].items():
                kind = cmeta["kind"]
                if kind == "array":
                    spec = specs[cmeta["buffer_idx"]]
                    block, _ = get_buf(spec)
                    arr = ShmArray.from_block(
                        block, shape=cmeta["shape"], dtype=np.dtype(cmeta["dtype"])
                    )
                    coords[cname] = (tuple(cmeta["dims"]), arr)
                elif kind == "pickle":
                    values = pickle.loads(cmeta["payload"])
                    coords[cname] = (tuple(cmeta["dims"]), values)
                else:
                    raise ValueError(f"Unknown coord kind: {kind!r}")

            # Rebuild data_vars
            for vname, vmeta in g_meta["data_vars"].items():
                spec = specs[vmeta["buffer_idx"]]
                block, _ = get_buf(spec)
                arr = ShmArray.from_block(
                    block, vmeta["shape"], dtype=np.dtype(vmeta["dtype"])
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
        buffers: list[ShmRef] = []
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
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        meta = payload["meta"]
        fields_meta: dict[str, Any] = meta["fields"]
        kwargs: dict[str, Any] = {}

        assert len(args) > 0
        pool = args[0]

        specs = payload["buffers"]

        for field_name, fmeta in fields_meta.items():
            codec_name = fmeta["codec"]
            start = fmeta["start"]
            count = fmeta["count"]

            subpayload: PayloadRef = {
                "codec": codec_name,
                "meta": fmeta["meta"],
                "buffers": specs[start : start + count],
            }

            # IMPORTANT: slice buffer_specs in the same way as buffers
            value = self._registry.decode(subpayload, pool)
            kwargs[field_name] = value

        return self._cls(**kwargs)
