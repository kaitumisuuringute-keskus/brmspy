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

    def alloc(self, size: int, temporary: bool = False) -> ShmBlock:
        print(f"alloc {'temp' if temporary else ''}")
        shm = self._manager.SharedMemory(size=size)
        block = ShmBlock(
            name=shm.name,
            size=shm.size,
            shm=shm,
            content_size=size,
            temporary=temporary,
        )
        if not temporary:
            self._blocks[block.name] = block
        return block

    def attach(self, ref: ShmRef) -> ShmBlock:
        if ref["name"] in self._blocks:
            return self._blocks[ref["name"]]
        shm = SharedMemory(name=ref["name"])
        block = ShmBlock(
            name=ref["name"],
            size=ref["size"],
            shm=shm,
            content_size=ref["content_size"],
            temporary=ref["temporary"],
        )
        if not ref["temporary"]:
            self._blocks[ref["name"]] = block
        return block

    def close_all(self) -> None:
        for block in self._blocks.values():
            block.shm.close()
        self._blocks.clear()

    def gc(self, name: str | None = None):
        if name is not None:
            b = self._blocks.pop(name, None)
            if b is not None:
                b.shm.close()
            return

        # Keep this code temporarily
        for key in list(self._blocks.keys()):
            b = self._blocks[key]
            if b.temporary:
                b.shm.close()
            del self._blocks[key]


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
        block = pool.attach(ref)
        if block.shm.buf is None:
            raise Exception("block.smh.buf is None!")
        blocks.append(block)
    return blocks
