from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from brmspy.types.session_types import ShmRef
from brmspy.types.shm import ShmBlock, ShmPool as _ShmPool


class ShmPool(_ShmPool):
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


def attach_buffers(pool: ShmPool, refs: list[ShmRef]) -> list[memoryview]:
    views: list[memoryview] = []
    for ref in refs:
        block = pool.attach(ref["name"], ref["size"])
        if block.shm.buf is None:
            raise Exception("block.smh.buf is None!")
        views.append(memoryview(block.shm.buf))
    return views
