"""
Dep installation tests for Windows, Ubuntu, and macOS

These tests are DESTRUCTIVE for the R environment.
DO NOT run locally!

These tests ONLY run within github actions, as running 
all 3 major platform images from a single local machine 
is both legally and technically difficult.
"""

import pytest
import pandas as pd
import numpy as np
import warnings

def _fit_minimal_model(brms):
    # run a very small model to verify the installation
    epilepsy = brms.get_brms_data("epilepsy")
    
    # Fit model (with reduced iterations for testing)
    model = brms.fit(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        iter=100,
        warmup=50,
        chains=2,
        silent=2,
        refresh=0
    )
    
    # Check it worked - now returns arviz InferenceData by default
    import arviz as az
    assert isinstance(model.idata, az.InferenceData)
    
    # Check key parameters exist
    param_names = list(model.idata.posterior.data_vars)
    assert any('b_zAge' in p for p in param_names)
    assert any('b_zBase' in p for p in param_names)


def _remove_deps():
    import rpy2.robjects as ro
    import rpy2.robjects.packages as rpackages
    import sys

    if rpackages.isinstalled("brms"):
        ro.r('remove.packages("brms")')
    if rpackages.isinstalled("cmdstanr"):
        ro.r('remove.packages("cmdstanr")')

    # since other tests might have imported brmspy already with global _brms singleton set,
    # we need to remove it from sys.modules first
    for name in list(sys.modules.keys()):
        if name.startswith("brmspy"):
            del sys.modules[name]


@pytest.mark.crossplatform
class TestCrossplatformInstall:
    """Test brms installation and version checking on 3 major OS."""
    
    @pytest.mark.slow
    def test_brms_install(self):
        import rpy2.robjects.packages as rpackages
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Keep brmspy after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.helpers.singleton import _get_brms
        with pytest.raises(ImportError):
            _get_brms()

        brms.install_brms(use_prebuilt_binaries=False)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = _get_brms()
        assert _brms is not None

        _fit_minimal_model(brms)
    
    @pytest.mark.slow
    def test_brms_install_prebuilt(self):
        import rpy2.robjects.packages as rpackages
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Keep brmspy after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.helpers.singleton import _get_brms
        with pytest.raises(ImportError):
            _get_brms()

        brms.install_brms(use_prebuilt_binaries=True)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = _get_brms()
        assert _brms is not None

        _fit_minimal_model(brms)
