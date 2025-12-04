"""
Tests for brmspy.binaries.env module (platform detection & compatibility).

Focus: Platform detection, toolchain compatibility, and error messages.

Note: these can be tested by running on real systems across
Linux, macOS, and Windows. The actual platform detection, R version checks,
and toolchain validation will be exercised naturally on different OSes.
"""

import pytest
