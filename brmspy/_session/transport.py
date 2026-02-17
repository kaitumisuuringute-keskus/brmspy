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
    """
    Concrete shared-memory pool implementation with slab allocation.

    When a slab is active (via `open_slab`/`seal_slab`), `alloc()` bump-allocates
    sub-regions from a single large SharedMemory block, drastically reducing the
    number of file descriptors for workloads that encode many small arrays
    (e.g. InferenceData with hundreds of data variables).

    _blocks dict keeps references to shm buffers TEMPORARILY and is cleaned up
    before each 'responding to main' or 'sending new message to worker'. This
    allows the in-between processing of shm buffers to rely on the buffers not
    being garbage collected.

    After reconstructing an object from a shm buffer, it's the CodecRegistrys role
    to take over the reference by initiating a weakref between the reconstructed
    object and buffer (or skipping if the object is temporary).
    """

    _ALIGN = 64  # cache-line alignment for sub-allocations

    def __init__(self, manager: SharedMemoryManager) -> None:
        self._manager = manager
        self._blocks: dict[str, ShmBlock] = {}
        # Slab state
        self._slab: SharedMemory | None = None
        self._slab_offset: int = 0
        self._slab_capacity: int = 0

    # -- slab API ----------------------------------------------------------

    def open_slab(self, capacity: int) -> None:
        """Pre-allocate a slab of at least *capacity* bytes.

        While a slab is open, `alloc()` bump-allocates from it instead of
        creating individual SharedMemory segments.  Call `seal_slab()` when
        the batch of allocations is complete.
        """
        self._slab = self._manager.SharedMemory(size=capacity)
        self._slab_offset = 0
        self._slab_capacity = self._slab.size
        # Track the slab itself so gc()/close_all() can manage it
        slab_block = ShmBlock(
            name=self._slab.name,
            size=self._slab.size,
            content_size=self._slab.size,
            shm=self._slab,
            temporary=False,
            offset=0,
        )
        self._blocks[self._slab.name] = slab_block

    def seal_slab(self) -> None:
        """Finalize the current slab (no more sub-allocations)."""
        self._slab = None
        self._slab_offset = 0
        self._slab_capacity = 0

    # -- allocation --------------------------------------------------------

    def alloc(self, size: int, temporary: bool = False) -> ShmBlock:
        if self._slab is not None:
            return self._alloc_from_slab(size, temporary)
        # No active slab — individual SharedMemory segment
        shm = self._manager.SharedMemory(size=size)
        block = ShmBlock(
            name=shm.name,
            size=shm.size,
            shm=shm,
            content_size=size,
            temporary=temporary,
            offset=0,
        )
        self._blocks[block.name] = block
        return block

    def _alloc_from_slab(self, size: int, temporary: bool) -> ShmBlock:
        align = self._ALIGN
        aligned = (self._slab_offset + align - 1) & ~(align - 1)

        if aligned + size > self._slab_capacity:
            # Current slab is full — start a new one sized for the remainder
            self.seal_slab()
            self.open_slab(max(size * 2, 4 * 1024 * 1024))
            aligned = 0

        block = ShmBlock(
            name=self._slab.name,
            size=size,
            content_size=size,
            shm=self._slab,
            temporary=temporary,
            offset=aligned,
        )
        self._slab_offset = aligned + size
        # Sub-blocks are NOT tracked in _blocks — the slab itself is tracked
        return block

    # -- attach / cleanup --------------------------------------------------

    def attach(self, ref: ShmRef) -> ShmBlock:
        name = ref["name"]
        offset = ref.get("offset", 0)
        # Reuse existing mapping for this SHM segment
        existing = self._blocks.get(name)
        if existing is not None:
            shm = existing.shm
        else:
            shm = SharedMemory(name=name)

        block = ShmBlock(
            name=name,
            size=ref["size"],
            shm=shm,
            content_size=ref["content_size"],
            temporary=ref["temporary"],
            offset=offset,
        )
        # Track the base mapping (only if new)
        if existing is None:
            self._blocks[name] = block
        return block

    def close_all(self) -> None:
        for block in self._blocks.values():
            block.shm.close()
        self._blocks.clear()
        self.seal_slab()

    def gc(self, name: str | None = None):
        if name is not None:
            b = self._blocks.pop(name, None)
            if b is not None:
                b.shm.close()
            return

        for key in list(self._blocks.keys()):
            b = self._blocks[key]
            if b.temporary:
                b.shm.close()
            del self._blocks[key]
