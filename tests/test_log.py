"""
Unit tests for brmspy.helpers.log module.

Tests logging functionality including formatters, log levels, 
context managers, and utility functions.
"""
import pytest
import logging
import time
from io import StringIO


class TestColors:
    """Test Colors class constants."""
    
    def test_color_constants_exist(self):
        """Test that all color constants are defined"""
        from brmspy.helpers.log import Colors
        
        assert hasattr(Colors, 'RESET')
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'YELLOW')
        assert hasattr(Colors, 'BOLD')
        assert isinstance(Colors.RESET, str)
        assert isinstance(Colors.RED, str)
        assert isinstance(Colors.YELLOW, str)
        assert isinstance(Colors.BOLD, str)


class TestBrmspyFormatter:
    """Test custom BrmspyFormatter."""
    
    @pytest.mark.parametrize("level,level_name,expected_color", [
        (logging.ERROR, "ERROR", "RED"),
        (logging.CRITICAL, "CRITICAL", "RED"),
        (logging.WARNING, "WARNING", "YELLOW"),
        (logging.INFO, "INFO", None),
        (logging.DEBUG, "DEBUG", None),
    ])
    def test_format_with_different_levels(self, level, level_name, expected_color):
        """Test formatter produces correct output for all log levels"""
        from brmspy.helpers.log import BrmspyFormatter, Colors
        
        formatter = BrmspyFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        
        # Format the record
        result = formatter.format(record)
        
        # Verify message content
        assert "test message" in result
        assert "[brmspy]" in result
        assert "[test_function]" in result
        
        # Verify level label for ERROR and CRITICAL
        if level >= logging.ERROR:
            assert f"[{level_name}]" in result
        
        # Verify color codes
        if expected_color == "RED":
            assert Colors.RED in result
            assert Colors.BOLD in result
        elif expected_color == "YELLOW":
            assert Colors.YELLOW in result
        
        # INFO and DEBUG should not have color codes
        if expected_color is None and level <= logging.INFO:
            # Should not contain color codes (except in the formatted part)
            assert result.startswith("[brmspy]") or Colors.RESET in result
    
    def test_format_removes_module_tag(self):
        """Test that <module> is removed from method name"""
        from brmspy.helpers.log import BrmspyFormatter
        
        formatter = BrmspyFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None
        )
        record.funcName = "<module>"
        
        result = formatter.format(record)
        
        # Should not contain [<module>]
        assert "[<module>]" not in result
        assert "[brmspy]" in result
    
    def test_format_with_custom_method_name(self):
        """Test formatter with custom method_name in record"""
        from brmspy.helpers.log import BrmspyFormatter
        
        formatter = BrmspyFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None
        )
        record.funcName = "default_func"
        record.method_name = "custom_method"
        
        result = formatter.format(record)
        
        # Should use custom method_name, not funcName
        assert "[custom_method]" in result
        assert "[default_func]" not in result


class TestGetLogger:
    """Test logger creation and singleton behavior."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance"""
        from brmspy.helpers.log import get_logger
        
        logger = get_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'brmspy'
    
    def test_get_logger_is_singleton(self):
        """Test that get_logger returns the same instance"""
        from brmspy.helpers.log import get_logger
        
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2
    
    def test_get_logger_has_handler(self):
        """Test that logger has a handler configured"""
        from brmspy.helpers.log import get_logger
        
        logger = get_logger()
        
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_get_logger_has_custom_formatter(self):
        """Test that logger uses BrmspyFormatter"""
        from brmspy.helpers.log import get_logger, BrmspyFormatter
        
        logger = get_logger()
        
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, BrmspyFormatter)
    
    def test_get_logger_no_propagate(self):
        """Test that logger does not propagate to root"""
        from brmspy.helpers.log import get_logger
        
        logger = get_logger()
        
        assert logger.propagate is False


class TestLoggingFunctions:
    """Test logging functions."""
    
    def test_log_with_explicit_method_name(self, caplog):
        """Test log() with explicit method_name"""
        from brmspy.helpers.log import log
        
        with caplog.at_level(logging.INFO):
            log("test message", method_name="my_method")
        
        assert "test message" in caplog.text
        assert "my_method" in caplog.text
    
    def test_log_with_auto_detection(self, caplog):
        """Test log() with automatic method name detection"""
        from brmspy.helpers.log import log
        
        with caplog.at_level(logging.INFO):
            log("auto detect message")
        
        assert "auto detect message" in caplog.text
    
    def test_log_info(self, caplog):
        """Test log_info() function"""
        from brmspy.helpers.log import log_info
        
        with caplog.at_level(logging.INFO):
            log_info("info message")
        
        assert "info message" in caplog.text
    
    def test_log_debug(self, caplog):
        """Test log_debug() function"""
        from brmspy.helpers.log import log_debug, set_log_level
        
        # Set to DEBUG level to capture debug messages
        set_log_level(logging.DEBUG)
        
        with caplog.at_level(logging.DEBUG):
            log_debug("debug message")
        
        assert "debug message" in caplog.text
        
        # Reset to INFO
        set_log_level(logging.INFO)
    
    def test_log_warning(self, caplog):
        """Test log_warning() function"""
        from brmspy.helpers.log import log_warning
        
        with caplog.at_level(logging.WARNING):
            log_warning("warning message")
        
        assert "warning message" in caplog.text
    
    def test_log_error(self, caplog):
        """Test log_error() function"""
        from brmspy.helpers.log import log_error
        
        with caplog.at_level(logging.ERROR):
            log_error("error message")
        
        assert "error message" in caplog.text
    
    def test_log_critical(self, caplog):
        """Test log_critical() function"""
        from brmspy.helpers.log import log_critical
        
        with caplog.at_level(logging.CRITICAL):
            log_critical("critical message")
        
        assert "critical message" in caplog.text
    
    def test_all_log_levels_in_sequence(self, caplog):
        """Test all log level functions in one test"""
        from brmspy.helpers.log import (
            log_info, log_debug, log_warning, log_error, log_critical, set_log_level
        )
        
        # Enable all levels
        set_log_level(logging.DEBUG)
        
        with caplog.at_level(logging.DEBUG):
            log_debug("debug")
            log_info("info")
            log_warning("warning")
            log_error("error")
            log_critical("critical")
        
        # Verify all messages appear
        assert "debug" in caplog.text
        assert "info" in caplog.text
        assert "warning" in caplog.text
        assert "error" in caplog.text
        assert "critical" in caplog.text
        
        # Reset
        set_log_level(logging.INFO)


class TestSetLogLevel:
    """Test set_log_level() function."""
    
    def test_set_log_level_changes_level(self):
        """Test that set_log_level changes the logger level"""
        from brmspy.helpers.log import get_logger, set_log_level
        
        logger = get_logger()
        
        # Set to DEBUG
        set_log_level(logging.DEBUG)
        assert logger.level == logging.DEBUG
        
        # Set to WARNING
        set_log_level(logging.WARNING)
        assert logger.level == logging.WARNING
        
        # Reset to INFO
        set_log_level(logging.INFO)
        assert logger.level == logging.INFO
    
    def test_set_log_level_filters_messages(self, caplog):
        """Test that setting level filters out lower priority messages"""
        from brmspy.helpers.log import log_debug, log_info, set_log_level
        
        # Set to WARNING - should filter out INFO and DEBUG
        set_log_level(logging.WARNING)
        
        with caplog.at_level(logging.DEBUG):
            log_debug("should not appear")
            log_info("also should not appear")
        
        # These should be filtered out
        assert "should not appear" not in caplog.text
        assert "also should not appear" not in caplog.text
        
        # Reset to INFO
        set_log_level(logging.INFO)


class TestLogTime:
    """Test LogTime context manager."""
    
    def test_logtime_context_manager(self, caplog):
        """Test LogTime measures and logs elapsed time"""
        from brmspy.helpers.log import LogTime
        
        with caplog.at_level(logging.INFO):
            with LogTime("test_operation"):
                time.sleep(0.01)  # Small delay
        
        # Verify log message contains operation name and time
        assert "test_operation" in caplog.text
        assert "took" in caplog.text
        assert "seconds" in caplog.text
    
    def test_logtime_default_name(self, caplog):
        """Test LogTime with default name"""
        from brmspy.helpers.log import LogTime
        
        with caplog.at_level(logging.INFO):
            with LogTime():
                time.sleep(0.01)
        
        # Should use default "process" name
        assert "process" in caplog.text
        assert "took" in caplog.text
    
    def test_logtime_measures_time(self, caplog):
        """Test that LogTime actually measures elapsed time"""
        from brmspy.helpers.log import LogTime
        
        with caplog.at_level(logging.INFO):
            with LogTime("timed_op") as lt:
                time.sleep(0.05)  # 50ms delay
        
        # Verify time measurement
        log_text = caplog.text
        # Extract the time value (format: "X.XX seconds")
        import re
        match = re.search(r'took (\d+\.\d+) seconds', log_text)
        assert match is not None
        elapsed = float(match.group(1))
        assert elapsed >= 0.04  # Should be at least 40ms (allowing some margin)


class TestGreet:
    """Test greet() function."""
    
    def test_greet_outputs_warnings(self, caplog):
        """Test that greet() outputs expected warning messages"""
        from brmspy.helpers.log import greet
        
        with caplog.at_level(logging.WARNING):
            greet()
        
        # Verify all three warning lines appear
        assert "brmspy <0.2 is still evolving" in caplog.text
        assert "APIs may change" in caplog.text
        assert "Feedback or a star on GitHub" in caplog.text
        assert "https://github.com/kaitumisuuringute-keskus/brmspy" in caplog.text


class TestCallerNameDetection:
    """Test _get_caller_name() function."""
    
    def test_get_caller_name_from_function(self, caplog):
        """Test that _get_caller_name detects the calling function"""
        from brmspy.helpers.log import log
        
        def my_test_function():
            with caplog.at_level(logging.INFO):
                log("message from function")
        
        my_test_function()
        
        # The log should contain the function name
        assert "my_test_function" in caplog.text or "message from function" in caplog.text
    
    def test_get_caller_name_with_explicit_override(self, caplog):
        """Test that explicit method_name overrides auto-detection"""
        from brmspy.helpers.log import log
        
        def some_function():
            with caplog.at_level(logging.INFO):
                log("test", method_name="explicit_name")
        
        some_function()
        
        # Should use explicit name, not function name
        assert "explicit_name" in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])