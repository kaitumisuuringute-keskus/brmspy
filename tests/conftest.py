"""
Pytest configuration and shared fixtures for brmspy tests
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



@pytest.fixture
def sample_dataframe():
    """
    Create a simple DataFrame for testing.
    
    Returns a small dataset with a continuous outcome and predictors.
    """
    np.random.seed(42)
    n = 50
    data = pd.DataFrame({
        'y': np.random.normal(10, 2, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.choice(['A', 'B'], n),
        'group': np.repeat(['G1', 'G2'], n//2)
    })
    return data


@pytest.fixture
def sample_dict():
    """
    Create a simple dictionary for testing data conversion.
    """
    return {
        'a': [1, 2, 3, 4, 5],
        'b': [2.1, 3.2, 4.3, 5.4, 6.5],
        'c': ['x', 'y', 'z', 'x', 'y']
    }


@pytest.fixture
def simple_formula():
    """Simple linear regression formula"""
    return "y ~ x1"


@pytest.fixture
def complex_formula():
    """Formula with random effects"""
    return "y ~ x1 + x2 + (1|group)"


@pytest.fixture
def poisson_data():
    """
    Create sample count data for Poisson regression testing.
    """
    np.random.seed(42)
    n = 40
    x = np.random.normal(0, 1, n)
    lambda_true = np.exp(1 + 0.5 * x)
    y = np.random.poisson(lambda_true)
    
    return pd.DataFrame({
        'count': y,
        'predictor': x
    })


def pytest_configure(config):
    """
    Custom pytest configuration.
    """
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring R and brms"
    )
    config.addinivalue_line(
        "markers",
        "requires_brms: marks tests that require brms to be installed"
    )
    config.addinivalue_line(
        "markers",
        "rdeps: rdeps test. only runs within githubs r-dependencies-tests workflow."
    )


@pytest.fixture(scope="session")
def brms_available():
    """
    Check if brms is available and can be imported.
    
    Returns True if brms is available, False otherwise.
    This is a session-scoped fixture that only checks once.
    """
    try:
        import rpy2.robjects.packages as rpackages
        brms = rpackages.importr("brms")
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests when required.
    """
    skip_requires_brms = pytest.mark.skip(reason="brms not installed - run: python -c 'import brmspy; brmspy.install_brms()'")
    skip_requires_rdeps = pytest.mark.skip(reason="rdeps test. only runs within githubs rdeps test workflow.'")
    skip_only_using_rdeps = pytest.mark.skip(reason="Running in rdeps-only mode!'")
    
    user_mark_expr = config.getoption("-m") or ""
    rdeps_allowed = (
        "rdeps" in user_mark_expr
        and os.getenv('BRMSPY_DESTRUCTIVE_RDEPS_TESTS') == "1"
    )

    # Try to check if brms is available
    brms_is_available = False
    try:
        import rpy2.robjects.packages as rpackages
        brms = rpackages.importr("brms")
        brms_is_available = True
    except:
        pass

    for item in items:
        if not brms_is_available and "requires_brms" in item.keywords:
            item.add_marker(skip_requires_brms)
        if not rdeps_allowed and "rdeps" in item.keywords:
            item.add_marker(skip_requires_rdeps)
        if rdeps_allowed and "rdeps" not in item.keywords:
            item.add_marker(skip_only_using_rdeps)
            
