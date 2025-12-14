from __future__ import annotations

"""
Session-layer types used by the main↔worker IPC protocol.

These are mostly structural types (TypedDict / dataclasses / Protocols) used by:

- the proxy session in [`brmspy._session.session`][brmspy._session.session]
- the worker loop in [`brmspy._session.worker.worker`][brmspy._session.worker.worker]
- codecs in [`brmspy._session.codec.base`][brmspy._session.codec.base]

The key invariant is that the main process must not hold live rpy2/R objects.
Instead, R objects are represented as lightweight handles (`SexpWrapper`) that
can be reattached inside the worker via its SEXP cache.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from brmspy.types.shm import ShmBlock, ShmBlockSpec, ShmRef

CommandType = Literal["CALL", "SHUTDOWN", "PING", "_RUN_TEST_BY_NAME"]


@dataclass
class SexpWrapper:
    """
    Lightweight handle for an R object stored in the worker.

    The worker keeps the real rpy2 `Sexp` in an internal cache and replaces it in
    results with this wrapper. When passed back to the worker, the wrapper is
    resolved to the original `Sexp` again.

    Notes
    -----
    - `SexpWrapper` instances are only meaningful within the lifetime of the
      worker process that produced them. After a worker restart, previously
      returned wrappers can no longer be reattached.
    - This type exists to keep the main process free of rpy2 / embedded-R state.
    """

    _rid: int
    _repr: str

    def __str__(self) -> str:
        return self._repr

    def __repr__(self) -> str:
        return self._repr


class PayloadRef(TypedDict):
    """
    Encoded argument/result payload sent over the control pipe.

    A payload is:

    - `codec`: the codec identifier used by the registry
    - `meta`: JSON-serializable metadata needed to reconstruct the value
    - `buffers`: shared-memory buffer references backing the payload
    """

    codec: str
    meta: dict[str, Any]
    buffers: list[ShmRef]


class Request(TypedDict):
    """
    IPC request message sent from main process to worker.

    Attributes
    ----------
    id : str
        Correlation id for the request/response pair.
    cmd : {"CALL", "SHUTDOWN", "PING", "_RUN_TEST_BY_NAME"}
        Command type.
    target : str
        Worker target spec (see [`_resolve_module_target()`][brmspy._session.worker.worker._resolve_module_target]).
    args, kwargs
        Encoded arguments.
    """

    id: str
    cmd: CommandType
    target: str
    args: list[PayloadRef]
    kwargs: dict[str, PayloadRef]


class Response(TypedDict):
    """
    IPC response message sent from worker back to the main process.
    """

    id: str
    ok: bool
    result: None | PayloadRef
    error: None | str
    traceback: None | str


@dataclass
class EnvironmentConfig:
    """
    Worker environment configuration.

    This configuration is applied in the worker before importing/using brms.

    Parameters
    ----------
    r_home : str or None
        Override for `R_HOME`. If None, the worker will rely on system detection.
    startup_scripts : list[str]
        R code snippets executed in the worker after initialization.
    environment_name : str
        brmspy environment name (used to determine `~/.brmspy/environment/<name>/Rlib`).
    runtime_path : str or None
        Path to a brmspy runtime bundle to activate in the worker.
    env : dict[str, str]
        Extra environment variables applied when spawning the worker.
    """

    r_home: None | str = None
    startup_scripts: list[str] = field(default_factory=list)
    environment_name: str = "default"
    runtime_path: None | str = None
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for persistence to JSON."""
        return {
            "environment_name": self.environment_name,
            "r_home": self.r_home,
            "startup_scripts": self.startup_scripts or [],
            "runtime_path": self.runtime_path,
            "env": self.env,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> EnvironmentConfig:
        """Deserialize configuration from a JSON object."""
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
        """Normalize `None | dict | EnvironmentConfig` into an `EnvironmentConfig`."""
        if obj is None:
            return cls()
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        return obj


@dataclass
class EncodeResult:
    """
    Result of encoding a Python value for IPC transfer.

    Attributes
    ----------
    codec : str
        Codec identifier.
    meta : dict[str, Any]
        JSON-serializable metadata required for decoding.
    buffers : list[ShmBlockSpec]
        Shared-memory blocks backing the encoded payload.
    """

    codec: str
    meta: dict[str, Any]
    buffers: list[ShmBlockSpec]


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol implemented by codecs in the session codec registry.
    """

    def can_encode(self, obj: Any) -> bool: ...

    def encode(self, obj: Any, shm_pool: Any) -> EncodeResult: ...

    def decode(
        self,
        meta: dict[str, Any],
        buffers: list[ShmBlock],
        buffer_specs: list[dict],
        shm_pool: Any,
    ) -> Any: ...
