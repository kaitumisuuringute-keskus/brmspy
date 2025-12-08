from __future__ import annotations

from typing import Any, Dict, List
import pickle

import numpy as np

from .base import Encoder, EncodeResult, ShmBlockSpec


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
