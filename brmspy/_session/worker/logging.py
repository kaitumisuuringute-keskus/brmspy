import builtins
import logging
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue
import os


def _print(*values: object, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(v) for v in values) + end
    logging.info(msg.rstrip("\n"))


if os.environ.get("BRMSPY_WORKER") == "1":
    builtins.print = _print


def setup_worker_logging(log_queue: Queue, level: int | None = None) -> None:
    """Route all worker stdout/stderr + logging into the parent's log queue."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level or logging.INFO)
    root.addHandler(QueueHandler(log_queue))
