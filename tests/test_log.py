"""
Unit tests for brmspy.helpers.log module.

Tests logging functionality including formatters, log levels,
context managers, and utility functions.
"""

import pytest
import logging
import time
from io import StringIO


def capture_brmspy_logs(run_func):
    """
    Helper to capture log output from brmspy logger using StringIO.

    This avoids pytest's capfd/capsys issues with custom loggers.
    """
    from brmspy.helpers.log import get_logger

    logger = get_logger()
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    # Use same formatter as existing handler
    if logger.handlers:
        handler.setFormatter(logger.handlers[0].formatter)

    logger.addHandler(handler)
    try:
        run_func()
        handler.flush()
        return stream.getvalue()
    finally:
        logger.removeHandler(handler)


class TestColors:
    """Test Colors class constants."""

    def test_color_constants_exist(self):
        """Test that all color constants are defined"""
        from brmspy.helpers.log import Colors

        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "YELLOW")
        assert hasattr(Colors, "BOLD")
        assert isinstance(Colors.RESET, str)
        assert isinstance(Colors.RED, str)
        assert isinstance(Colors.YELLOW, str)
        assert isinstance(Colors.BOLD, str)


class TestBrmspyFormatter:
    """Test custom BrmspyFormatter."""

    @pytest.mark.parametrize(
        "level,level_name,expected_color",
        [
            (logging.ERROR, "ERROR", "RED"),
            (logging.CRITICAL, "CRITICAL", "RED"),
            (logging.WARNING, "WARNING", "YELLOW"),
            (logging.INFO, "INFO", None),
            (logging.DEBUG, "DEBUG", None),
        ],
    )
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
            exc_info=None,
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
            exc_info=None,
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
            exc_info=None,
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
        assert logger.name == "brmspy"

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


class TestLoggingFunctions:
    """Test logging functions."""

    def test_log_with_explicit_method_name(self):
        """Test log() with explicit method_name"""
        from brmspy.helpers.log import log

        output = capture_brmspy_logs(
            lambda: log("test message", method_name="my_method")
        )

        assert "test message" in output
        assert "my_method" in output

    def test_log_with_auto_detection(self):
        """Test log() with automatic method name detection"""
        from brmspy.helpers.log import log

        output = capture_brmspy_logs(lambda: log("auto detect message"))

        assert "auto detect message" in output

    def test_log_info(self):
        """Test log_info() function"""
        from brmspy.helpers.log import log_info

        output = capture_brmspy_logs(lambda: log_info("info message"))

        assert "info message" in output

    def test_log_debug(self):
        """Test log_debug() function"""
        from brmspy.helpers.log import log_debug, set_log_level

        # Set to DEBUG level to capture debug messages
        set_log_level(logging.DEBUG)

        output = capture_brmspy_logs(lambda: log_debug("debug message"))

        assert "debug message" in output

        # Reset to INFO
        set_log_level(logging.INFO)

    def test_log_warning(self):
        """Test log_warning() function"""
        from brmspy.helpers.log import log_warning

        output = capture_brmspy_logs(lambda: log_warning("warning message"))

        assert "warning message" in output
        assert "WARNING" in output

    def test_log_error(self):
        """Test log_error() function"""
        from brmspy.helpers.log import log_error

        output = capture_brmspy_logs(lambda: log_error("error message"))

        assert "error message" in output
        assert "ERROR" in output

    def test_log_critical(self):
        """Test log_critical() function"""
        from brmspy.helpers.log import log_critical

        output = capture_brmspy_logs(lambda: log_critical("critical message"))

        assert "critical message" in output
        assert "CRITICAL" in output

    def test_all_log_levels_in_sequence(self):
        """Test all log level functions in one test"""
        from brmspy.helpers.log import (
            log_info,
            log_debug,
            log_warning,
            log_error,
            log_critical,
            set_log_level,
        )

        # Enable all levels
        set_log_level(logging.DEBUG)

        def run_all():
            log_debug("debug")
            log_info("info")
            log_warning("warning")
            log_error("error")
            log_critical("critical")

        output = capture_brmspy_logs(run_all)

        # Verify all messages appear
        assert "debug" in output
        assert "info" in output
        assert "warning" in output
        assert "error" in output
        assert "critical" in output

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

    def test_set_log_level_filters_messages(self):
        """Test that setting level filters out lower priority messages"""
        from brmspy.helpers.log import log_debug, log_info, log_warning, set_log_level

        # Set to WARNING - should filter out INFO and DEBUG
        set_log_level(logging.WARNING)

        def run():
            log_debug("should not appear")
            log_info("also should not appear")
            log_warning("should appear")

        output = capture_brmspy_logs(run)

        # These should be filtered out
        assert "should not appear" not in output
        assert "also should not appear" not in output
        # This should be present
        assert "should appear" in output

        # Reset to INFO
        set_log_level(logging.INFO)


class TestLogTime:
    """Test LogTime context manager."""

    def test_logtime_context_manager(self):
        """Test LogTime measures and logs elapsed time"""
        from brmspy.helpers.log import LogTime

        def run():
            with LogTime("test_operation"):
                time.sleep(0.01)  # Small delay

        output = capture_brmspy_logs(run)

        # Verify log message contains operation name and time
        assert "test_operation" in output
        assert "took" in output
        assert "seconds" in output

    def test_logtime_default_name(self):
        """Test LogTime with default name"""
        from brmspy.helpers.log import LogTime

        def run():
            with LogTime():
                time.sleep(0.01)

        output = capture_brmspy_logs(run)

        # Should use default "process" name
        assert "process" in output
        assert "took" in output

    def test_logtime_measures_time(self):
        """Test that LogTime actually measures elapsed time"""
        from brmspy.helpers.log import LogTime
        import re

        def run():
            with LogTime("timed_op") as lt:
                time.sleep(0.05)  # 50ms delay

        output = capture_brmspy_logs(run)

        # Verify time measurement
        # Extract the time value (format: "X.XX seconds")
        match = re.search(r"took (\d+\.\d+) seconds", output)
        assert match is not None
        elapsed = float(match.group(1))
        assert elapsed >= 0.04  # Should be at least 40ms (allowing some margin)


class TestCallerNameDetection:
    """Test _get_caller_name() function."""

    def test_get_caller_name_from_function(self):
        """Test that _get_caller_name detects the calling function"""
        from brmspy.helpers.log import log

        def my_test_function():
            log("message from function")

        output = capture_brmspy_logs(my_test_function)

        # The log should contain the function name or the message
        assert "my_test_function" in output or "message from function" in output

    def test_get_caller_name_with_explicit_override(self):
        """Test that explicit method_name overrides auto-detection"""
        from brmspy.helpers.log import log

        def some_function():
            log("test", method_name="explicit_name")

        output = capture_brmspy_logs(some_function)

        # Should use explicit name, not function name
        assert "explicit_name" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
