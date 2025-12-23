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


class RWorkerCrashedError(RuntimeError):
    """
    Raised when the R worker process crashes during an operation.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    recovered : bool
        Indicates whether a fresh worker session was successfully started.

        * ``True``  – The crash occurred, but automatic recovery succeeded.
                      The failed operation did *not* complete, but the worker
                      is now in a clean state. Callers may safely retry.
        * ``False`` – The crash occurred and automatic recovery failed.
                      A usable worker session is not available. Callers should
                      treat this as a hard failure and abort or escalate.
    cause : BaseException, optional
        The original exception that triggered the crash. Stored as ``__cause__``
        for chained exception inspection.

    Usage
    -----
    In user code or automated pipelines, you can distinguish between a
    recoverable and unrecoverable crash:

    ```python
    try:
        brms.brm(...)
    except RWorkerCrashedError as err:
        if err.recovered:
            # Crash occurred, but a fresh worker is ready.
            # Safe to retry the operation once.
            brms.brm(...)
        else:
            # Worker could not be restarted.
            # Treat this as a hard failure.
            raise
    ```

    Notes
    -----
    All crashes automatically produce a new exception that wraps the original
    failure using Python's exception chaining (``raise ... from cause``).
    Inspect ``err.__cause__`` for the underlying system error.
    """

    def __init__(
        self, message: str, *, recovered: bool, cause: BaseException | None = None
    ):
        super().__init__(message)
        self.recovered = recovered
        self.__cause__ = cause
