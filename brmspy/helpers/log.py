import inspect
import logging
import sys


# --- filters ---------------------------------------------------------


def _running_under_pytest() -> bool:
    return (
        "PYTEST_CURRENT_TEST" in os.environ  # reliable with pytest >=3
        or "pytest" in sys.modules
    )


class PrintOnlyFilter(logging.Filter):
    """Allow only records that came from our print() override."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "from_print", False)


class NonPrintFilter(logging.Filter):
    """Block records that came from print()."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "from_print", False)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"


# Custom formatter that adds the `[brmspy][method_name]` prefix with colors
class BrmspyFormatter(logging.Formatter):
    """
    Custom formatter that formats log messages as [brmspy][method_name] msg.
    Adds color coding for warnings (yellow) and errors (red) when terminal supports it.
    """

    def format(self, record):
        # Get method name from record or use the function name
        method_name = getattr(record, "method_name", record.funcName)

        if record.levelno >= logging.ERROR:
            # Red color for errors and critical
            level_label = "ERROR" if record.levelno == logging.ERROR else "CRITICAL"
            prefix = f"{Colors.RED}{Colors.BOLD}[brmspy][{method_name}][{level_label}]{Colors.RESET}"
        elif record.levelno == logging.WARNING:
            # Yellow color for warnings
            prefix = f"{Colors.YELLOW}[brmspy][{method_name}][WARNING]{Colors.RESET}"
        else:
            # No color for info and debug
            prefix = f"[brmspy][{method_name}]"

        prefix = prefix.replace("[<module>]", "")

        # Format the message with the custom prefix
        original_format = self._style._fmt
        self._style._fmt = f"{prefix} %(message)s"

        result = super().format(record)

        # Restore original format
        self._style._fmt = original_format

        return result


# Create and configure the logger
_logger = None


def get_logger() -> logging.Logger:
    """
    Get or create the brmspy logger instance.

    Returns a configured logger with a custom formatter that outputs
    messages in the format: `[brmspy][method_name] msg here`

    Returns
    -------
    logging.Logger
        Configured brmspy logger instance

    Examples
    --------
    >>> from brmspy.helpers.log import get_logger
    >>> logger = get_logger()
    >>> logger.info("Starting process")  # Prints: [brmspy][<module>] Starting process
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger("brmspy")
        _logger.setLevel(logging.INFO)

        if not _logger.handlers:
            # Handler for "normal" logs
            normal_handler = logging.StreamHandler()
            normal_handler.setFormatter(BrmspyFormatter())
            normal_handler.addFilter(NonPrintFilter())
            _logger.addHandler(normal_handler)

            # print logs: preserve control chars and explicit \n/\r
            print_handler = logging.StreamHandler()
            print_handler.setFormatter(logging.Formatter("%(message)s"))
            print_handler.addFilter(PrintOnlyFilter())
            print_handler.terminator = ""
            _logger.addHandler(print_handler)

        if _running_under_pytest():
            _logger.propagate = True
        else:
            _logger.propagate = False

    return _logger


def _get_caller_name() -> str:
    """
    Get the name of the calling function/method.

    Returns
    -------
    str
        Name of the calling function or "unknown" if not found
    """
    frame = inspect.currentframe()
    if frame is not None:
        try:
            # Go back 3 frames: this function -> log() -> log_info/log_warning/etc -> actual caller
            caller_frame = frame.f_back
            if caller_frame is not None:
                caller_frame = caller_frame.f_back
                if caller_frame is not None:
                    caller_frame = caller_frame.f_back
                    if caller_frame is not None:
                        return caller_frame.f_code.co_name
        finally:
            del frame
    return "unknown"


def log(*msg: str, method_name: str | None = None, level: int = logging.INFO):
    """
    Log a message with automatic method name detection.

    Parameters
    ----------
    msg : str
        The message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.
    level : int, optional
        Logging level (default: logging.INFO)
    """
    if method_name is None:
        method_name = _get_caller_name()

    msg_str = " ".join(str(v) for v in msg)

    logger = get_logger()
    logger.log(level, msg_str, extra={"method_name": method_name})


def log_info(msg: str, method_name: str | None = None):
    """
    Log an info message.

    Parameters
    ----------
    msg : str
        The message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.
    """
    log(msg, method_name=method_name, level=logging.INFO)


def log_debug(msg: str, method_name: str | None = None):
    """
    Log a debug message.

    Parameters
    ----------
    msg : str
        The message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.

    """
    log(msg, method_name=method_name, level=logging.DEBUG)


def log_warning(msg: str, method_name: str | None = None):
    """
    Log a warning message.

    Parameters
    ----------
    msg : str
        The warning message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.

    """
    log(msg, method_name=method_name, level=logging.WARNING)


def log_error(msg: str, method_name: str | None = None):
    """
    Log an error message.

    Parameters
    ----------
    msg : str
        The error message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.
    """
    log(msg, method_name=method_name, level=logging.ERROR)


def log_critical(msg: str, method_name: str | None = None):
    """
    Log a critical message.

    Parameters
    ----------
    msg : str
        The critical message to log
    method_name : str, optional
        The name of the method/function. If None, will auto-detect from call stack.
    """
    log(msg, method_name=method_name, level=logging.CRITICAL)


def set_log_level(level: int):
    """
    Set the logging level for brmspy logger.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    logger = get_logger()
    logger.setLevel(level)


import os
import time


class LogTime:
    def __init__(self, name="process"):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        log(f"[{self.name}] took {elapsed:.2f} seconds")
