import logging
import inspect
from typing import Optional


# ANSI color codes
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'


# Custom formatter that adds the [brmspy][method_name] prefix with colors
class BrmspyFormatter(logging.Formatter):
    """
    Custom formatter that formats log messages as [brmspy][method_name] msg.
    Adds color coding for warnings (yellow) and errors (red) when terminal supports it.
    """
    
    def format(self, record):
        # Get method name from record or use the function name
        method_name = getattr(record, 'method_name', record.funcName)
        
        # Determine prefix based on log level
        if record.levelno >= logging.ERROR:
            # Red color for errors and critical
            level_label = 'ERROR' if record.levelno == logging.ERROR else 'CRITICAL'
            prefix = f'{Colors.RED}{Colors.BOLD}[brmspy][{method_name}][{level_label}]{Colors.RESET}'
        elif record.levelno == logging.WARNING:
            # Yellow color for warnings
            prefix = f'{Colors.YELLOW}[brmspy][{method_name}][WARNING]{Colors.RESET}'
        else:
            # No color for info and debug
            prefix = f'[brmspy][{method_name}]'
        
        prefix = prefix.replace("[<module>]", "")
        
        # Format the message with the custom prefix
        original_format = self._style._fmt
        self._style._fmt = f'{prefix} %(message)s'
        
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
    messages in the format: [brmspy][method_name] msg here
    
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
        _logger = logging.getLogger('brmspy')
        _logger.setLevel(logging.INFO)
        
        # Only add handler if none exists (avoid duplicate handlers)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(BrmspyFormatter())
            _logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
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


def log(msg: str, method_name: Optional[str] = None, level: int = logging.INFO):
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
    
    logger = get_logger()
    logger.log(level, msg, extra={'method_name': method_name})


def log_info(msg: str, method_name: Optional[str] = None):
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


def log_debug(msg: str, method_name: Optional[str] = None):
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


def log_warning(msg: str, method_name: Optional[str] = None):
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


def log_error(msg: str, method_name: Optional[str] = None):
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


def log_critical(msg: str, method_name: Optional[str] = None):
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


def greet():
    log_warning("brmspy <0.2 is still evolving; APIs may change.")
    log_warning("Feedback or a star on GitHub helps guide development:")
    log_warning("https://github.com/kaitumisuuringute-keskus/brmspy")