import logging
import inspect
from typing import Optional


# Custom formatter that adds the [brmspy][method_name] prefix
class BrmspyFormatter(logging.Formatter):
    """
    Custom formatter that formats log messages as [brmspy][method_name] msg.
    """
    
    def format(self, record):
        # Get method name from record or use the function name
        method_name = getattr(record, 'method_name', record.funcName)
        
        # Format the message with the custom prefix
        original_format = self._style._fmt
        self._style._fmt = f'[brmspy][{method_name}] %(message)s'
        
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log
    >>> 
    >>> def my_function():
    ...     log("Starting process")  # Prints: [brmspy][my_function] Starting process
    >>> 
    >>> def custom_name():
    ...     log("Custom name", method_name="custom")  # Prints: [brmspy][custom] Custom name
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log_info
    >>> 
    >>> def my_function():
    ...     log_info("Processing data")  # Prints: [brmspy][my_function] Processing data
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log_debug
    >>> 
    >>> def my_function():
    ...     log_debug("Debug info")  # Prints: [brmspy][my_function] Debug info
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log_warning
    >>> 
    >>> def my_function():
    ...     log_warning("This might be an issue")  # Prints: [brmspy][my_function] This might be an issue
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log_error
    >>> 
    >>> def my_function():
    ...     log_error("Something went wrong")  # Prints: [brmspy][my_function] Something went wrong
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
    
    Examples
    --------
    >>> from brmspy.helpers.log import log_critical
    >>> 
    >>> def my_function():
    ...     log_critical("Critical failure")  # Prints: [brmspy][my_function] Critical failure
    """
    log(msg, method_name=method_name, level=logging.CRITICAL)


def set_log_level(level: int):
    """
    Set the logging level for brmspy logger.
    
    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    
    Examples
    --------
    >>> from brmspy.helpers.log import set_log_level
    >>> import logging
    >>> 
    >>> # Enable debug logging
    >>> set_log_level(logging.DEBUG)
    >>> 
    >>> # Disable all but error messages
    >>> set_log_level(logging.ERROR)
    """
    logger = get_logger()
    logger.setLevel(level)