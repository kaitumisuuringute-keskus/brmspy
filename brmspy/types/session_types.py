from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from brmspy.types.shm import ShmBlockSpec, ShmRef

CommandType = Literal["CALL", "SHUTDOWN"]


@dataclass
class SexpWrapper:
    _rid: int
    _repr: str

    def __str(self):
        return self._repr

    def __repr__(self):
        return self._repr


class PayloadRef(TypedDict):
    codec: str
    meta: dict[str, Any]
    buffers: list[ShmRef]


class Request(TypedDict):
    id: str
    cmd: CommandType
    target: str
    args: list[PayloadRef]
    kwargs: dict[str, PayloadRef]


class Response(TypedDict):
    id: str
    ok: bool
    result: None | PayloadRef
    error: None | str
    traceback: None | str


@dataclass
class EnvironmentConfig:
    r_home: None | str = None
    startup_scripts: list[str] = field(default_factory=list)
    environment_name: str = "default"
    runtime_path: None | str = None
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_name": self.environment_name,
            "r_home": self.r_home,
            "startup_scripts": self.startup_scripts or [],
            "runtime_path": self.runtime_path,
            "env": self.env,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> EnvironmentConfig:
        return cls(
            r_home=obj["r_home"],
            startup_scripts=obj["startup_scripts"],
            environment_name=obj["environment_name"],
            runtime_path=obj["runtime_path"],
            env=obj["env"],
        )

    @classmethod
    def from_obj(
        cls, obj: None | dict[str, Any] | EnvironmentConfig
    ) -> EnvironmentConfig:
        if obj is None:
            return cls()
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        return obj


@dataclass
class EncodeResult:
    codec: str
    meta: dict[str, Any]
    buffers: list[ShmBlockSpec]


@runtime_checkable
class Encoder(Protocol):
    def can_encode(self, obj: Any) -> bool: ...

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult: ...

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[memoryview],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any: ...
