"""
Worker-process implementation package (internal).

This package contains the spawned worker entrypoint and worker-only helpers
(logging redirection, embedded-R setup, Sexp cache).
"""

from .worker import worker_main

__all__ = ["worker_main"]
