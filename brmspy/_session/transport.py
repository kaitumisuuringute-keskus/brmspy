"""
Shared-memory transport utilities (internal).

`RModuleSession` uses shared memory to move large payloads between main and worker.
The parent allocates blocks and passes only `(name, size)` references over the Pipe.
The worker (or the main process during decode) attaches by name to access buffers.

This module implements the concrete `ShmPool` used by the session layer.
"""

from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

from brmspy.types.session import ShmRef
from brmspy.types.shm import ShmBlock
from brmspy.types.shm import ShmPool as _ShmPool


class ShmPool(_ShmPool):
    """Concrete shared-memory pool implementation that tracks attached blocks."""

    def __init__(self, manager: SharedMemoryManager) -> None:
        self._manager = manager
        self._blocks: dict[str, ShmBlock] = {}

    def alloc(self, size: int) -> ShmBlock:
        shm = self._manager.SharedMemory(size=size)
        block = ShmBlock(name=shm.name, size=size, shm=shm)
        self._blocks[block.name] = block
        return block

    def attach(self, name: str, size: int) -> ShmBlock:
        shm = SharedMemory(name=name)
        block = ShmBlock(name=name, size=size, shm=shm)
        self._blocks[name] = block
        return block

    def close_all(self) -> None:
        for block in self._blocks.values():
            block.shm.close()
        self._blocks.clear()


def attach_buffers(pool: ShmPool, refs: list[ShmRef]) -> list[ShmBlock]:
    """
    Attach to a list of SHM blocks and return their `memoryview`s.

    Parameters
    ----------
    pool : ShmPool
        Pool used for attaching blocks by name.
    refs : list[brmspy.types.session.ShmRef]
        List of `(name, size)` references.

    Returns
    -------
    list[memoryview]
        Views over each shared-memory buffer.
    """
    blocks: list[ShmBlock] = []
    for ref in refs:
        block = pool.attach(ref["name"], ref["size"])
        if block.shm.buf is None:
            raise Exception("block.smh.buf is None!")
        blocks.append(block)
    return blocks
