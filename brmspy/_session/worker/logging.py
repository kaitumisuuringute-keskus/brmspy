"""
Worker-side logging/printing integration (internal).

The parent process owns the main logging configuration. The worker routes its own
logging records and `print()` output into the parent's log queue, so users see a
single unified stream.
"""

import builtins
import logging
import os
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue

from brmspy.helpers.log import get_logger


def setup_worker_logging(log_queue: Queue, level: int | None = None) -> None:
    """
    Configure worker logging to forward into the parent's log queue.

    Parameters
    ----------
    log_queue : multiprocessing.queues.Queue
        Queue owned by the parent; the worker will emit `logging` records into it.
    level : int | None, optional
        Root log level for the worker process. Defaults to `logging.INFO`.

    Notes
    -----
    When `BRMSPY_WORKER=1`, the worker replaces [`builtins.print`](https://docs.python.org/3/library/functions.html#print)
    to preserve raw control characters and line endings produced by R/cmdstan.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level or logging.INFO)
    root.addHandler(QueueHandler(log_queue))

    logger = get_logger()

    def _print(*values: object, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")

        # Preserve raw control chars and end exactly as R/cmdstan intended
        msg = sep.join(str(v) for v in values) + end

        if msg == "":
            return

        logger.info(
            msg,
            extra={
                "method_name": "_print",
                "no_prefix": True,
                "from_print": True,  # important for filters
            },
        )

    if os.environ.get("BRMSPY_WORKER") == "1":
        builtins.print = _print
