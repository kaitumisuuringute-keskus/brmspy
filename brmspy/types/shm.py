from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import TypedDict


class ShmRef(TypedDict):
    name: str
    size: int


@dataclass
class ShmBlockSpec:
    name: str
    size: int


@dataclass
class ShmBlock(ShmBlockSpec):
    shm: SharedMemory


class ShmPool:
    def __init__(self, manager: SharedMemoryManager) -> None: ...

    def alloc(self, size: int) -> ShmBlock: ...

    def attach(self, name: str, size: int) -> ShmBlock: ...

    def close_all(self) -> None: ...
