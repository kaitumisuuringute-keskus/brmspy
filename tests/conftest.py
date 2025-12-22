"""
Pytest configuration and shared fixtures for brmspy tests
"""

import inspect
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["BRMSPY_TEST"] = "1"


@pytest.fixture(autouse=True)
def _force_brmspy_cleanup_between_tests():
    """
    Forcefully clean up brmspy worker + SHM between tests.

    Rationale (macOS CI):
    - POSIX shared memory attaches (`/psm_*`) consume file descriptors.
    - If SHM blocks accumulate across tests, the process can hit `[Errno 24] Too many open files`.
    - This fixture ensures each test ends with a full worker teardown, releasing SHM FDs.
    """
    yield

    # Never run cleanup logic inside the worker process itself.
    if os.environ.get("BRMSPY_WORKER") == "1":
        return

    try:
        from brmspy import brms

        if getattr(brms, "_is_rsession", True):
            try:
                # Restart between tests to force-close worker-side `/psm_*` SHM FDs.
                cast(Any, brms).restart(empty_shm=True)
            except Exception:
                pass
    except Exception:
        pass

    # Encourage timely finalizers/GC for objects holding SHM-backed views.
    import gc

    gc.collect()


@pytest.fixture
def sample_dataframe():
    """
    Create a simple DataFrame for testing.

    Returns a small dataset with a continuous outcome and predictors.
    """
    np.random.seed(42)
    n = 50
    data = pd.DataFrame(
        {
            "y": np.random.normal(10, 2, n),
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.choice(["A", "B"], n),
            "group": np.repeat(["G1", "G2"], n // 2),
        }
    )
    return data


@pytest.fixture
def sample_dict():
    """
    Create a simple dictionary for testing data conversion.
    """
    return {
        "a": [1, 2, 3, 4, 5],
        "b": [2.1, 3.2, 4.3, 5.4, 6.5],
        "c": ["x", "y", "z", "x", "y"],
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

    return pd.DataFrame({"count": y, "predictor": x})


def pytest_configure(config):
    """
    Custom pytest configuration.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring R and brms"
    )
    config.addinivalue_line(
        "markers", "requires_brms: marks tests that require brms to be installed"
    )
    config.addinivalue_line(
        "markers",
        "rdeps: rdeps test. only runs within githubs r-dependencies-tests workflow.",
    )
    config.addinivalue_line(
        "markers", "worker: run this test inside the brms worker process"
    )


@pytest.fixture(scope="session")
def brms_available():
    """
    Check if brms is available and can be imported.

    Returns True if brms is available, False otherwise.
    This is a session-scoped fixture that only checks once.
    """
    try:
        from brmspy import brms

        return brms.get_brms_version() is not None
    except Exception:
        return False


class ExplodingRobjects:
    """
    Sentinel object put in sys.modules["rpy2.robjects*"].

    Any *test* code that tries to use rpy2.robjects on the main process
    will immediately hard-fail. Import machinery can still probe dunders.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    def __getattr__(self, name: str):
        # let import/system internals probe dunders without noise
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        raise RuntimeError(
            f"rpy2.robjects usage is forbidden in tests on the main process "
            f"(in {self._label}, tried to access attribute {name!r})"
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"rpy2.robjects usage is forbidden in tests on the main process "
            f"(in {self._label}, tried to call it like a function)"
        )


@pytest.fixture(autouse=True, scope="module")
def no_robjects_bomb():

    if os.environ.get("BRMSPY_WORKER") == "1":
        yield
        return

    mp = pytest.MonkeyPatch()

    # Module-level import: `import rpy2.robjects as ro`
    mp.setitem(sys.modules, "rpy2.robjects", ExplodingRobjects("rpy2.robjects"))
    mp.setitem(
        sys.modules,
        "rpy2.robjects.packages",
        ExplodingRobjects("rpy2.robjects.packages"),
    )
    mp.setitem(
        sys.modules,
        "rpy2.robjects.vectors",
        ExplodingRobjects("rpy2.robjects.vectors"),
    )

    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def worker_runner():
    """Returns a function that runs a test fn inside the worker."""
    from brmspy import brms
    from brmspy._session.session import RModuleSession

    session = cast(RModuleSession, brms)

    def run_remote(module, cls, func):
        return session._run_test_by_name(module, cls, func)

    return run_remote


def _nodeid_to_target(nodeid: str) -> tuple[str, str | None, str]:
    """
    Convert pytest nodeid:
        tests/test_file.py::TestClass::test_name
    into:
        tests.test_file, TestClass, test_name
    """
    parts = nodeid.split("::")
    file_part = parts[0].replace("\\", "/")  # just in case
    if file_part.endswith(".py"):
        file_part = file_part[:-3]
    mod_path = file_part.replace("/", ".")

    if len(parts) >= 3:
        class_name = parts[-2]
        func_name = parts[-1]
    else:
        class_name = None
        func_name = parts[-1]

    return mod_path, class_name, func_name


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """
    Intercept only @pytest.mark.worker tests and run them remotely.
    Returning True tells pytest "do not run locally".
    """
    if "worker" not in pyfuncitem.keywords:
        return None  # run normally

    # For now: forbid pytest fixtures in worker tests (until you implement fixture shipping)
    argnames = tuple(getattr(pyfuncitem, "_fixtureinfo").argnames)
    if argnames:
        pytest.fail(
            f"@pytest.mark.worker tests can't use pytest fixtures yet (found {argnames}).",
            pytrace=False,
        )

    run_remote = pyfuncitem._request.getfixturevalue("worker_runner")

    mod_path, class_name, func_name = _nodeid_to_target(pyfuncitem.nodeid)

    # should raise on failure (and pytest will mark FAIL)
    run_remote(mod_path, class_name, func_name)

    return True


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests when required.
    """
    skip_requires_brms = pytest.mark.skip(
        reason="brms not installed - run: python -c 'import brmspy; brmspy.install_brms()'"
    )
    skip_requires_rdeps = pytest.mark.skip(
        reason="rdeps test. only runs within githubs rdeps test workflow.'"
    )
    skip_only_using_rdeps = pytest.mark.skip(reason="Running in rdeps-only mode!'")

    user_mark_expr = config.getoption("-m") or ""
    rdeps_allowed = (
        "rdeps" in user_mark_expr and os.getenv("BRMSPY_DESTRUCTIVE_RDEPS_TESTS") == "1"
    )

    # Try to check if brms is available
    brms_is_available = False
    try:
        from brmspy import brms

        if not rdeps_allowed:
            name = "_test"
            if not brms.environment_exists(name):
                with brms.manage(environment_name=name) as ctx:
                    ctx.install_brms(use_prebuilt=True)
            else:
                brms.environment_activate(name)

        if brms.get_brms_version() is not None:
            brms_is_available = True
    except Exception as e:
        print(f"Failed to set up brms {e}")
        pass

    for item in items:
        if not brms_is_available and "requires_brms" in item.keywords:
            item.add_marker(skip_requires_brms)
        if not rdeps_allowed and "rdeps" in item.keywords:
            item.add_marker(skip_requires_rdeps)
        if rdeps_allowed and "rdeps" not in item.keywords:
            item.add_marker(skip_only_using_rdeps)
