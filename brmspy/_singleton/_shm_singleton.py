from ..types.shm import ShmPool

_shm: ShmPool | None = None


def _get_shm() -> ShmPool | None:
    global _shm
    return _shm


def _set_shm(shm: ShmPool | None):
    global _shm
    _shm = shm


__all__ = ["_get_shm", "_set_shm"]
