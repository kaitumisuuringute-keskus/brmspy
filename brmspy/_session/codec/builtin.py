from __future__ import annotations
import json

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
    ShmSeriesMetadata,
    ShmArray,
    ShmDataFrameColumns,
    ShmDataFrameSimple,
)
from ...types.shm import ShmBlock, ShmRef

ONE_MB = 1024 * 1024


class NumpyArrayCodec(Encoder):
    """SHM-backed codec for [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)."""

    def can_encode(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        _, ref, dtype, shape, order = ShmArray.to_shm(obj, shm_pool)
        return EncodeResult(
            codec=type(self).__name__,
            meta={"dtype": dtype, "shape": shape, "order": order},
            buffers=[ref],
        )

    def decode(
        self,
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        buf, _ = get_buf(payload["buffers"][0])

        return ShmArray.from_metadata(payload["meta"], buf)


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
            "index": pickle.dumps(obj.index, protocol=pickle.HIGHEST_PROTOCOL),
            "variant": "single",
        }
        buffers: list[ShmRef] = []

        if obj.empty:
            meta["variant"] = "empty"
        elif isinstance(obj, ShmDataFrameSimple):
            # single dtype matrix
            meta["variant"] = "single"
            meta["dtype"] = str(obj.values.dtype)
            meta["order"] = ShmArray.array_order(obj.values)
            buffers.append(obj._shm_metadata)
        elif isinstance(obj, ShmDataFrameColumns):
            # per column buffers
            meta["variant"] = "columnar"
            meta["order"] = "F"
            meta["columns"] = obj._shm_metadata
        else:
            # Fallback: put each column in its own SHM block
            meta["variant"] = "columnar"
            meta["order"] = "C"
            columns: dict[str, ShmSeriesMetadata] = {}

            for col_name in obj.columns:
                col = obj[col_name]

                arr_modified, block, dtype, shape, order = ShmArray.to_shm(
                    col, shm_pool
                )

                spec = ShmRef(
                    name=block["name"],
                    size=block["size"],
                    content_size=block["content_size"],
                )
                columns[col_name] = ShmDataFrameColumns._create_col_metadata(
                    obj[col_name], spec, arr_modified
                )
            meta["columns"] = columns

        return EncodeResult(codec=type(self).__name__, meta=meta, buffers=buffers)

    def decode(
        self,
        payload: PayloadRef,
        get_buf: Callable[[ShmRef], tuple[ShmBlock, memoryview]],
        *args,
    ) -> Any:
        meta = payload["meta"]
        if meta.get("variant") == "empty":
            return pd.DataFrame({})

        buffer_specs = payload["buffers"]

        index = pickle.loads(meta["index"])

        if meta.get("variant") == "single":
            spec = buffer_specs[0]
            buf, memview = get_buf(buffer_specs[0])
            dtype = np.dtype(meta["dtype"])
            nbytes = spec["size"]
            order = meta["order"]

            columns = meta["columns"]
            shape = (len(index), len(columns))

            # Only use the slice that actually holds array data
            view = memview[:nbytes]
            arr = np.ndarray(shape=shape, dtype=dtype, buffer=view, order=order)

            df = ShmDataFrameSimple(data=arr, index=index, columns=columns)
            df._set_shm_metadata(spec)

            return df
        elif meta.get("variant") == "columnar":
            columns_metadata: dict[str, ShmSeriesMetadata] = meta["columns"]
            nrows = len(index)

            columns = list(columns_metadata.keys())

            data: dict[str, pd.Series] = {}

            for i, col_name in enumerate(columns):
                metadata = columns_metadata[col_name]
                spec = metadata["block"]
                buf, view = get_buf(spec)
                data[col_name] = ShmDataFrameColumns._reconstruct_series(
                    metadata, buf, nrows, index
                )

            df = ShmDataFrameColumns(data=data)
            df._set_shm_metadata(columns_metadata)

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
                _, spec, dtype, shape, order = ShmArray.to_shm(arr, shm_pool)
                meta = {"dtype": dtype, "shape": shape, "order": order}
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
