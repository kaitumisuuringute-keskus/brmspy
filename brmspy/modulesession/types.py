from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Literal, Dict, Any, List, Optional

CommandType = Literal["CALL", "SHUTDOWN"]

@dataclass
class SexpWrapper:
    _rid: int
    _repr: str

    def __str(self):
        return self._repr

    def __repr__(self):
        return self._repr


class ShmRef(TypedDict):
    name: str
    size: int


class PayloadRef(TypedDict):
    codec: str
    meta: Dict[str, Any]
    buffers: List[ShmRef]


class Request(TypedDict):
    id: str
    cmd: CommandType
    target: str
    args: List[PayloadRef]
    kwargs: Dict[str, PayloadRef]


class Response(TypedDict):
    id: str
    ok: bool
    result: Optional[PayloadRef]
    error: Optional[str]
    traceback: Optional[str]
