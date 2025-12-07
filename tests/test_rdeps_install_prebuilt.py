"""
Dep installation tests for Windows, Ubuntu, and macOS

These tests are DESTRUCTIVE for the R environment.
DO NOT run locally!

These tests ONLY run within github actions, as running 
all 3 major platform images from a single local machine 
is both legally and technically difficult.
"""

import pytest



@pytest.mark.rdeps
class TestInstallPrebuilt:
    """Test brms installation and version checking on 3 major OS."""
    
    @pytest.mark.slow
    def test_brms_install_prebuilt(self):
        from brmspy import brms
        import rpy2.robjects.packages as rpackages
        from install_helpers import _remove_deps
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Import after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.runtime._state import get_brms
        

        brms.install_brms(use_prebuilt=True, install_rtools=False)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = get_brms()
        assert _brms is not None

        from install_helpers import _fit_minimal_model
        _fit_minimal_model(brms)

        # Use runtime API for deactivation
        from brmspy import runtime
        runtime.deactivate_runtime()
