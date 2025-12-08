from __future__ import annotations

from typing import TypedDict, Literal, Dict, Any, List, Optional

CommandType = Literal["CALL", "SHUTDOWN"]


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
