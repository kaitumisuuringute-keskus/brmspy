from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Literal, Dict, Any, List, Optional, Union

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


@dataclass
class EnvironmentConfig:
    r_home: Optional[str] = None
    startup_scripts: List[str] = field(default_factory=list)
    environment_name: str = "default"
    runtime_path: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment_name": self.environment_name,
            "r_home": self.r_home,
            "startup_scripts": self.startup_scripts or [],
            "runtime_path": self.runtime_path,
            "env": self.env
        }
    
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'EnvironmentConfig':
        return cls(
            r_home=obj['r_home'],
            startup_scripts=obj['startup_scripts'],
            environment_name=obj['environment_name'],
            runtime_path=obj['runtime_path'],
            env=obj['env']
        )
    
    @classmethod
    def from_obj(cls, obj: Optional[Union[Dict[str, Any], 'EnvironmentConfig']]) -> 'EnvironmentConfig':
        if obj is None:
            return cls()
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        return obj

