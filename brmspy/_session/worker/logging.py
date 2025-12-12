import builtins
import logging
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue
import os
from brmspy.helpers.log import get_logger


def setup_worker_logging(log_queue: Queue, level: int | None = None) -> None:
    """Route all worker stdout/stderr + logging into the parent's log queue."""
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

        # You can still drop truly empty messages if you want
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
        import builtins

        builtins.print = _print
