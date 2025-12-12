"""
Dep installation tests for Windows, Ubuntu, and macOS

These tests are DESTRUCTIVE for the R environment.
DO NOT run locally!

These tests ONLY run within github actions, as running
all 3 major platform images from a single local machine
is both legally and technically difficult.
"""

import uuid
import pytest


@pytest.mark.rdeps
class TestInstall:
    """Test brms installation and version checking on 3 major OS."""

    @pytest.mark.slow
    def test_brms_install(self):
        from brmspy import brms
        from install_helpers import _fit_minimal_model

        # Import after removal to ensure the library imports without brms installed
        from brmspy import brms

        env_name = "_test-" + uuid.uuid4().hex[:16]

        with brms.manage(environment_name=env_name) as ctx:
            ctx.install_brms(use_prebuilt=False)

        _brms = brms.get_brms_version()
        assert _brms is not None
        from install_helpers import _fit_minimal_model

        _fit_minimal_model(brms)
