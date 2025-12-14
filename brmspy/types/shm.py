from __future__ import annotations

"""
Shared-memory helper types used by the session transport.

The main↔worker transport uses `multiprocessing.shared_memory` for large payloads.
The parent allocates blocks and passes only small references (`name`, `size`)
over the control pipe; the worker attaches to the same blocks by name.

These types are intentionally lightweight and are used by:

- [`brmspy._session.transport`][brmspy._session.transport] (implementation of `ShmPool`)
- [`brmspy._session.codec.builtin`][brmspy._session.codec.builtin] (codecs that store arrays/frames in SHM)
- IPC message shapes in [`brmspy.types.session`][brmspy.types.session]
"""

from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import TypedDict


class ShmRef(TypedDict):
    """
    Reference to a shared-memory block sent over IPC.

    Attributes
    ----------
    name : str
        Shared memory block name (as assigned by `SharedMemoryManager`).
    size : int
        Allocated block size in bytes.

    Notes
    -----
    Codecs may store a logical payload smaller than `size`. In that case, the
    codec metadata must include the logical `nbytes`/length so that decoders can
    slice the buffer appropriately.
    """

    name: str
    size: int


@dataclass
class ShmBlockSpec:
    """
    Lightweight shared-memory block description.

    Parameters
    ----------
    name : str
        Shared memory block name.
    size : int
        Allocated size in bytes.
    """

    name: str
    size: int


@dataclass
class ShmBlock(ShmBlockSpec):
    """
    Attached shared-memory block (name/size + live `SharedMemory` handle).

    Notes
    -----
    This object owns a `SharedMemory` handle and must be closed when no longer
    needed. In brmspy this is managed by a `ShmPool` implementation.
    """

    shm: SharedMemory


class ShmPool:
    """
    Minimal interface for allocating and attaching shared-memory blocks.

    The concrete implementation lives in
    [`brmspy._session.transport.ShmPool`][brmspy._session.transport.ShmPool] and tracks
    blocks so they can be closed on teardown.
    """

    def __init__(self, manager: SharedMemoryManager) -> None:
        """
        Create a pool bound to an existing `SharedMemoryManager`.

        Parameters
        ----------
        manager : multiprocessing.managers.SharedMemoryManager
            Manager used to allocate blocks.
        """
        ...

    def alloc(self, size: int) -> ShmBlock:
        """
        Allocate a new shared-memory block.

        Parameters
        ----------
        size : int
            Size in bytes.

        Returns
        -------
        ShmBlock
            Newly allocated block.
        """
        ...

    def attach(self, name: str, size: int) -> ShmBlock:
        """
        Attach to an existing shared-memory block by name.

        Parameters
        ----------
        name : str
            Shared memory block name.
        size : int
            Allocated block size in bytes.

        Returns
        -------
        ShmBlock
            Attached block.
        """
        ...

    def close_all(self) -> None:
        """
        Close all tracked shared-memory handles owned by this pool.

        Returns
        -------
        None
        """
        ...
