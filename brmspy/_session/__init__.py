"""
Internal main↔worker session layer.

This package implements the process-isolated architecture described in
`docs/development/development.md`:

- main process exposes a proxy module surface
- worker process embeds R and executes calls
- large payloads move via shared memory and a codec registry

Most users should not import anything from here directly; the public entry point
is `brmspy.brms`, which wires an `RModuleSession` internally.
"""

from .session import RModuleSession

__all__ = ["RModuleSession"]
