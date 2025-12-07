# Development Guide

This guide covers the development infrastructure, build processes, and CI/CD architecture for brmspy.

## Quick Start

### Setup

```bash
# Clone and setup
git clone https://github.com/kaitumisuuringute-keskus/brmspy.git
cd brmspy

# Install with dev dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[all]"

# Install R dependencies (fast with prebuilt runtime)
python -c "import brmspy; brmspy.install_brms(use_prebuilt=True)"

# Or traditional installation from source (~30 minutes)
# python -c "import brmspy; brmspy.install_brms()"
```

## Project Architecture

### Directory Structure

```
brmspy/
├── brmspy/                    # Main package
│   ├── brms.py               # Core API exports
│   ├── types.py              # Type definitions and result dataclasses
│   ├── brms_functions/       # Modular brms function wrappers
│   │   ├── brm.py            # Model fitting (fit, brm)
│   │   ├── diagnostics.py    # Model diagnostics (summary, loo, etc.)
│   │   ├── families.py       # Family specifications
│   │   ├── formula.py        # Formula construction
│   │   ├── generic.py        # Generic function caller
│   │   ├── io.py             # Data I/O (save_rds, read_rds, etc.)
│   │   ├── prediction.py     # Predictions (posterior_predict, etc.)
│   │   ├── prior.py          # Prior specifications
│   │   └── stan.py           # Stan code generation
│   ├── runtime/              # Runtime management system (layered architecture)
│   │   ├── __init__.py       # Public API: install(), activate(), deactivate(), status()
│   │   ├── _types.py         # Foundation: RuntimeStatus, RuntimeManifest, SystemInfo
│   │   ├── _platform.py      # Foundation: Platform detection, fingerprinting
│   │   ├── _manifest.py      # Foundation: Manifest reading
│   │   ├── _r_packages.py    # R Layer: Package operations
│   │   ├── _r_env.py         # R Layer: Environment management
│   │   ├── _config.py        # Disk Layer: Config management (~/.brmspy/config.json)
│   │   ├── _storage.py       # Disk Layer: Runtime storage (~/.brmspy/runtime/)
│   │   ├── _download.py      # Disk Layer: Download utilities
│   │   ├── _github.py        # Disk Layer: GitHub API integration
│   │   ├── _rtools.py        # Disk Layer: Windows Rtools installation
│   │   ├── _state.py         # Orchestration: State management (env snapshots)
│   │   ├── _activation.py    # Orchestration: Runtime activation/deactivation
│   │   └── _install.py       # Orchestration: Installation logic
│   └── helpers/              # Internal utilities
│       ├── conversion.py     # Python ↔ R ↔ ArviZ
│       ├── log.py            # Logging utilities
│       ├── priors.py         # Prior builders
│       ├── robject_iter.py   # R object iteration
│       ├── rtools.py         # Rtools utilities (legacy)
│       └── singleton.py      # R package caching
├── .github/workflows/        # CI/CD pipelines
├── .runtime_builder/         # Docker for Linux builds
├── docs/                     # mkdocs documentation
└── tests/                    # Test suite
```

### Core Components

**brmspy/brms.py** - Main module that exports all public functions
**brmspy/types.py** - Type definitions and result dataclasses (FitResult, SummaryResult, LooResult, etc.)
**brmspy/brms_functions/** - Modular organization of brms function wrappers:
  - **brm.py** - Model fitting functions
  - **diagnostics.py** - 8 diagnostic functions (summary, fixef, ranef, loo, loo_compare, validate_newdata, etc.)
  - **prediction.py** - Prediction functions (posterior_predict, posterior_epred, posterior_linpred, log_lik)
  - **prior.py** - Prior specification functions
  - **families.py** - Family specifications and wrappers
  - **formula.py** - Formula construction helpers
  - **generic.py** - Generic function caller for unwrapped brms functions
  - **io.py** - Data I/O functions
  - **stan.py** - Stan code generation

**brmspy/runtime/** - Runtime management system with layered architecture (see below)
**brmspy/helpers/** - Internal conversion and utility functions

### Data Flow

```
Python Code
    ↓
brmspy.fit() [brms.py]
    ↓
Type Conversions [helpers/conversion.py]
    ↓
R brms via rpy2
    ↓
CmdStan MCMC Sampling
    ↓
ArviZ InferenceData
    ↓
Python Result Objects
```

## Runtime System Architecture

The runtime system provides prebuilt bundles to skip lengthy R package compilation. The new architecture uses a **layered design pattern** with strict separation of concerns.

### Layered Architecture

The runtime system is organized into 4 layers, each with specific responsibilities:

#### 1. Foundation Layer (No Dependencies)
Pure functions for system information and data structures.

- **`_types.py`** - Type definitions: [`RuntimeStatus`](../brmspy/runtime/_types.py), [`RuntimeManifest`](../brmspy/runtime/_types.py), [`SystemInfo`](../brmspy/runtime/_types.py)
- **`_platform.py`** - Platform detection, system fingerprinting, compatibility checks
- **`_manifest.py`** - Manifest file reading and validation

#### 2. R Layer (Depends on Foundation)
R package operations and environment management.

- **`_r_packages.py`** - Install/unload R packages, build CmdStan
- **`_r_env.py`** - Manage R environment variables, GitHub token forwarding

#### 3. Disk Layer (Depends on Foundation)
File system operations and external integrations.

- **`_config.py`** - Persistent config at [`~/.brmspy/config.json`](../brmspy/runtime/_config.py)
- **`_storage.py`** - Runtime storage at [`~/.brmspy/runtime/`](../brmspy/runtime/_storage.py)
- **`_download.py`** - Download and hash verification utilities
- **`_github.py`** - GitHub API for release discovery
- **`_rtools.py`** - Windows Rtools installation

#### 4. Orchestration Layer (Depends on All Lower Layers)
High-level workflows coordinating multiple components.

- **`_state.py`** - Environment snapshot/restore for activation
- **`_activation.py`** - Runtime activation/deactivation logic
- **`_install.py`** - Installation orchestration (traditional + prebuilt)

#### Public API (`__init__.py`)
Four clean functions for users:

```python
from brmspy import runtime

# Install R packages traditionally (~30 min) or prebuilt (~1 min)
runtime.install(use_prebuilt=True)

# Activate a specific runtime
runtime.activate(runtime_path)

# Deactivate and restore original environment
runtime.deactivate()

# Query current state
status = runtime.status()
print(status.system.fingerprint)  # e.g., 'linux-x86_64-r4.5'
```

### System Fingerprint

Each runtime is identified by: `{os}-{arch}-r{major}.{minor}`

Examples:
- `linux-x86_64-r4.5`
- `macos-arm64-r4.5` (Apple Silicon)
- `windows-x86_64-r4.5`

Generated by [`_platform.system_fingerprint()`](../brmspy/runtime/_platform.py:52)

### Bundle Structure

```
brmspy-runtime-{fingerprint}-{version}.tar.gz
├── manifest.json              # Metadata (fingerprint, versions, build info)
├── cmdstan/                   # Compiled CmdStan binaries
└── Rlib/                      # R libraries (brms, cmdstanr, dependencies)
```

Installed to: `~/.brmspy/runtime/{fingerprint}-{version}/`

### Usage Examples

```python
import brmspy

# Fast installation with prebuilt runtime (~1 minute)
brmspy.install_brms(use_prebuilt=True)

# Traditional installation (~30 minutes)
brmspy.install_brms()

# Check current runtime status
status = brmspy.runtime.status()
print(f"Active: {status.active_runtime}")
print(f"System: {status.system.fingerprint}")
print(f"Can use prebuilt: {status.can_use_prebuilt}")

# Manual activation/deactivation
brmspy.runtime.activate("/path/to/runtime")
brmspy.runtime.deactivate()
```

**Note:** Runtime building is handled internally by CI/CD. Users should use [`install(use_prebuilt=True)`](../brmspy/runtime/__init__.py:236) rather than building locally.

## CI/CD Pipelines

All workflows in `.github/workflows/`:

### 1. Python Test Matrix (`python-test-matrix.yml`)

**Trigger:** Push/PR to master  
**Purpose:** Test Python 3.10, 3.12, 3.14 on Linux

**Workflow:**
1. Build CmdStan once (cached)
2. Test matrix in parallel
3. Update coverage badge (3.12 only)

**Key Features:**
- Shared R/CmdStan cache
- Parallel execution
- Coverage reporting

### 2. R Dependencies Tests (`r-dependencies-tests.yml`)

**Trigger:** Push/PR to master  
**Purpose:** Test on Linux, macOS, Windows

**Workflow:**
- Python 3.12 only
- Tests marked with `@pytest.mark.rdeps`
- Fail-fast disabled

### 3. Documentation (`docs.yml`)

**Trigger:** Push to master  
**Purpose:** Deploy docs to GitHub Pages

**Stack:**
- mkdocstrings for API docs
- Auto-deploys to https://kaitumisuuringute-keskus.github.io/brmspy/

### 4. PyPI Publish (`python-publish.yml`)

**Trigger:** GitHub Release created  
**Purpose:** Publish to PyPI

**Workflow:**
1. Run full test suite
2. Build: `python -m build`
3. Upload: `twine upload dist/*`

**Requirements:** `PYPI_USERNAME`, `PYPI_PASSWORD` secrets

### 5. Runtime Publish (`runtime-publish.yml`)

**Trigger:** Manual dispatch
**Purpose:** Build prebuilt runtimes for all platforms

**Architecture:**
1. Create GitHub Release (tag: `runtime`)
2. Build runtimes in parallel (Linux in Docker, macOS/Windows native)
3. Upload to release with attestation

**Linux Build (Docker):**
```yaml
- Pull: ghcr.io/.../brmspy-runtime-builder:ubuntu18-gcc9
- Install R 4.5.0
- Build runtime using internal build script
- Upload tarball
```

**macOS/Windows Build (Native):**
```yaml
- Setup Python 3.12 + R 4.5
- Install dependencies
- Build runtime using internal build script
- Upload tarball
```

**Note:** Runtime building uses internal tooling. The public API for users is [`brmspy.runtime.install(use_prebuilt=True)`](../brmspy/runtime/__init__.py).

### 6. Linux Runtime Builder (`build-linux-runtime-image.yml`)

**Trigger:** Manual dispatch  
**Purpose:** Build Docker image for Linux runtime compilation

**Image:** Ubuntu 18.04 + GCC 9 + Python 3.12 (for old glibc compatibility)

## Runtime Builder (`.runtime_builder/linux/`)

### Dockerfile

Creates build environment:
- **Base:** Ubuntu 18.04 (glibc 2.27)
- **Toolchain:** GCC 9, g++ 9, gfortran 9
- **Python:** 3.12.7 (compiled from source)
- **Dependencies:** BLAS, LAPACK, V8, GLPK, graphics libs

### install_r.sh

Smart R installation:
1. Try APT (fast)
2. Fallback to source compilation if version unavailable

### publish.sh

Builds and pushes Docker image to GHCR:
```bash
docker build -t ghcr.io/{owner}/brmspy-runtime-builder:{tag}
docker push ghcr.io/{owner}/brmspy-runtime-builder:{tag}
```

## Testing

### Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures (sample_dataframe, etc.)
├── test_basic.py                    # Basic functionality tests
├── test_diagnostics.py              # Diagnostics functions tests (14 tests)
├── test_families.py                 # Family specifications tests
├── test_generic.py                  # Generic function caller tests
├── test_integration.py              # End-to-end integration tests
├── test_io.py                       # I/O functions tests
├── test_predictions.py              # Prediction functions tests
├── test_priors.py                   # Prior specification tests
├── test_conversion.py               # Type conversion tests
├── test_log.py                      # Logging utility tests
├── test_rdeps_1_install.py          # Runtime installation tests (marked @pytest.mark.rdeps)
├── test_rdeps_build.py              # Runtime building tests
├── test_rdeps_config.py             # Config management tests
├── test_rdeps_github.py             # GitHub API tests
├── test_rdeps_install_extended.py   # Extended installation tests
├── test_runtime_r_env.py            # R environment management tests
└── test_runtime_storage.py          # Runtime storage tests
```

**Test Coverage:**
- **14 diagnostics tests** covering summary, fixef, ranef, posterior_summary, prior_summary, loo, loo_compare, validate_newdata
- **2 generic function tests** for call() wrapper
- All tests use `iter=100, warmup=50` for fast CI execution
- Tests marked with `@pytest.mark.slow` and `@pytest.mark.requires_brms`

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v --cov=brmspy      # With coverage
pytest -m rdeps                    # DESTRUCTIVE rdeps tests
pytest -n auto                     # Parallel (requires pytest-xdist)
```

### Test Markers

```python
@pytest.mark.rdeps
def test_basic_fit():
    """Runs on all platforms in CI"""
    pass
```

## Build and Release

### Version Management

Update in:
- `pyproject.toml`
- `settings.ini`
- `brmspy/__init__.py`

### Release Process

1. **Update versions** and CHANGELOG.md
2. **Test:** `pytest`
4. **Build:** `make build`
5. **Create GitHub Release** (tag: `release-0...`)
6. **CI automatically** tests and publishes to PyPI

### Building Runtimes

**Via GitHub Actions (Recommended):**
1. Go to **Actions** → **runtime-publish**
2. **Run workflow** with version and tag
3. Runtimes published to: `releases/tag/runtime`

**Local Development:**
Runtime building for local development uses internal scripts not exposed in the public API. For testing, use:

```python
# Install from source for testing
import brmspy
brmspy.install_brms()  # Traditional installation

# Or test with prebuilt from GitHub
brmspy.install_brms(use_prebuilt=True)
```

## Documentation

### mkdocs Configuration

**File:** `mkdocs.yml`

```yaml
site_name: brmspy
theme:
  name: shadcn
plugins:
  ...
```

### Docstring Style

All docstrings use **NumPy style** with `` ```python `` code blocks (no `.. code-block::`):

```python
def example(param: str) -> dict:
    """
    One-line summary.

    Detailed description.

    Parameters
    ----------
    param : str
        Parameter description

    Returns
    -------
    dict
        Return description

    Examples
    --------
    Basic usage:

    ```python
    result = example("hello")
    print(result)
    ```
    """
    return {"param": param}
```

## Performance

### R Package Caching

**Singleton pattern** in `brmspy/helpers/singleton.py`:

```python
from brmspy.helpers.singleton import get_r_package

brms = get_r_package("brms")  # First call: imports
brms = get_r_package("brms")  # Cached, instant
```

### Prebuilt Runtimes

| Method | Installation Time |
|--------|------------------|
| From source | 20-30 minutes |
| Prebuilt runtime | 20-60 seconds |

## Troubleshooting

### R Package Installation Fails

```bash
# Check R version (need 4.0+)
R --version

# Manual install
R -e "install.packages(c('cmdstanr', 'brms', 'posterior'))"
```

### CmdStan Compilation Fails

**Linux:**
```bash
sudo apt-get install build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
```python
import brmspy
# Automatically installs Rtools if needed
brmspy.install_brms(install_rtools=True)
```

Or manually via the runtime API:
```python
from brmspy.runtime import _rtools
_rtools.ensure_installed()  # Internal API
```

### Runtime Incompatibility

```python
# Check compatibility
from brmspy import runtime
status = runtime.status()
print(f"Can use prebuilt: {status.can_use_prebuilt}")
print(f"Issues: {status.compatibility_issues}")

# Install matching prebuilt (auto-detects platform)
import brmspy
brmspy.install_brms(use_prebuilt=True)

# Or install from source
brmspy.install_brms()  # Traditional installation
```

## Contributing

### Code Style

- **Docstrings:** NumPy style
- **Type hints:** Required for public APIs

### PR Process

1. Fork and create feature branch
2. Make changes and add tests
3. Run: `make format && make lint && make test`
4. Commit with conventional commits format
5. Open PR with clear description

## Resources

- **Documentation:** https://kaitumisuuringute-keskus.github.io/brmspy/
- **Repository:** https://github.com/kaitumisuuringute-keskus/brmspy
- **Issues:** https://github.com/kaitumisuuringute-keskus/brmspy/issues
- **PyPI:** https://pypi.org/project/brmspy/