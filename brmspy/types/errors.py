from __future__ import annotations

"""
Error types exposed by brmspy.

The main process talks to an isolated worker process that hosts embedded R.
When the worker reports an error, brmspy raises `RSessionError` in the main
process and may attach a `remote_traceback` captured in the worker.

See Also
--------
[`RModuleSession._decode_result()`][brmspy._session.session.RModuleSession._decode_result]
    Converts worker responses into Python return values or raises `RSessionError`.
[`worker_main()`][brmspy._session.worker.worker.worker_main]
    Worker loop that captures exceptions and sends structured error responses.
"""


class RSessionError(RuntimeError):
    """
    Error raised when a worker call fails.

    Parameters
    ----------
    message : str
        Human-readable error message (often derived from R error messages).
    remote_traceback : str or None, default=None
        Best-effort traceback text from the worker process. For R errors this may
        be an R traceback string; for Python errors inside the worker it may be
        a Python traceback.

    Notes
    -----
    This exception type is designed to preserve the *remote* failure context
    while keeping the main process free of rpy2/R state.
    """

    def __init__(self, message: str, remote_traceback: str | None = None) -> None:
        super().__init__(message)
        self.remote_traceback = remote_traceback

    def __str__(self) -> str:
        """Return message plus the remote traceback (if available)."""
        base = super().__str__()
        if self.remote_traceback:
            return f"{base}\n\nRemote traceback:\n{self.remote_traceback}\n\n"
        return base
