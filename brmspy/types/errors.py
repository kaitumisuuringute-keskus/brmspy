from __future__ import annotations


class RSessionError(RuntimeError):
    def __init__(self, message: str, remote_traceback: str | None = None) -> None:
        super().__init__(message)
        self.remote_traceback = remote_traceback

    def __str__(self) -> str:
        base = super().__str__()
        if self.remote_traceback:
            return f"{base}\n\nRemote traceback:\n{self.remote_traceback}\n\n"
        return base
