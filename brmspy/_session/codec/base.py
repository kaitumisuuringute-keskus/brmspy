from __future__ import annotations
from contextlib import contextmanager
import weakref


"""
Codec registry primitives (internal).

The session layer uses a codec registry to serialize Python values across the
main↔worker boundary. Encoders may allocate shared-memory blocks for large payloads
and return small metadata + SHM references for IPC transport.
"""

from dataclasses import is_dataclass
from typing import Any, Iterator

from brmspy.types.session import Encoder, EncodeResult, PayloadRef
from brmspy.types.shm import ShmBlock, ShmRef


def _noop(_blocks):
    pass


class CodecRegistry:
    """Ordered registry of encoders used for IPC serialization."""

    def __init__(self) -> None:
        self._by_codec: dict[str, Encoder] = {}
        self._encoders: list[Encoder] = []

    def register(self, encoder: Encoder) -> None:
        """
        Register an encoder instance.

        Parameters
        ----------
        encoder : brmspy.types.session.Encoder
            Encoder to register. Its `codec` attribute is used as the key when present,
            otherwise the class name is used.
        """
        if hasattr(encoder, "codec") and encoder.codec:  # type: ignore
            codec_name = encoder.codec  # type: ignore
        else:
            codec_name = type(encoder).__name__
        self._by_codec[codec_name] = encoder
        encoder.codec = codec_name  # type: ignore
        self._encoders.append(encoder)

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult:
        """
        Encode an object by selecting the first encoder that accepts it.

        Parameters
        ----------
        obj : Any
            Value to encode.
        shm_pool : Any
            SHM pool used by codecs for allocating buffers.

        Returns
        -------
        brmspy.types.session.EncodeResult
        """
        for enc in self._encoders:
            if enc.can_encode(obj):
                res = enc.encode(obj, shm_pool)
                if not res.codec:
                    res.codec = type(enc).__name__
                return res

        # fallback to pickle
        if "PickleCodec" not in self._by_codec:
            raise RuntimeError("No pickle codec registered")
        return self._by_codec["PickleCodec"].encode(obj, shm_pool)

    def decode(
        self,
        payload: PayloadRef,
        shm_pool: Any,
    ) -> Any:
        """
        Decode a payload using a named codec.

        Parameters
        ----------
        codec : str
            Codec identifier previously returned by `encode()`.
        meta : dict[str, Any]
            Codec metadata.
        buffers : list[memoryview]
            Memoryviews for attached SHM buffers.
        buffer_specs : list[dict]
            Original buffer specs (name/size) corresponding to `buffers`.
        shm_pool : Any
            SHM pool (some codecs may attach additional buffers).

        Returns
        -------
        Any
        """
        codec = payload["codec"]
        if codec not in self._by_codec:
            raise ValueError(
                f"Unknown codec: {codec}, available: {list(self._by_codec.keys())}"
            )

        buffers = []

        @contextmanager
        def get_buf(ref: ShmRef) -> Iterator[tuple[ShmBlock, memoryview]]:
            buf = shm_pool.attach(ref)
            memview = memoryview(buf.shm.buf)
            view = memview[: ref["content_size"]].cast("B")

            try:
                if not ref["temporary"]:
                    # non-temporary buffers are associated with columns / objects
                    buffers.append(buf)

                yield buf, view

            finally:
                # deterministic cleanup for temporary buffers
                if ref["temporary"]:
                    # IMPORTANT: release view before closing shm
                    try:
                        view.release()
                        memview.release()
                        shm_pool.gc(ref["name"])
                    except:
                        pass

        value = self._by_codec[codec].decode(payload, get_buf, shm_pool)
        self._attach_shm_lifetime(value, buffers)

        return value

    @classmethod
    def _attach_shm_lifetime(cls, obj: Any, shms: list[ShmBlock]) -> None:
        """Keep SHM blocks alive as long as `obj` is alive."""
        if not shms:
            return
        if obj is None or isinstance(obj, (bool, str, int, float)):
            return

        try:
            weakref.finalize(obj, _noop, tuple(shms))
        except:
            return
