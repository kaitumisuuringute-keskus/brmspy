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

# Install R dependencies
python -c "import brmspy; brmspy.install_brms()"
```

## Project Architecture

### Directory Structure

```
brmspy/
├── brmspy/                    # Main package
│   ├── brms.py               # Core API (fit, predict, etc.)
│   ├── types.py              # Type definitions
│   ├── install.py            # R dependency installation
│   ├── binaries/             # Prebuilt runtime system
│   │   ├── build.py          # Create runtime bundles
│   │   ├── env.py            # Platform detection
│   │   └── use.py            # Install runtimes
│   └── helpers/              # Internal utilities
│       ├── conversion.py     # Python ↔ R ↔ ArviZ
│       ├── priors.py         # Prior builders
│       ├── rtools.py         # Windows Rtools
│       └── singleton.py      # R package caching
├── .github/workflows/        # CI/CD pipelines
├── .runtime_builder/         # Docker for Linux builds
├── docs/                     # mkdocs documentation
└── tests/                    # Test suite
```

### Core Components

**brmspy/brms.py** - High-level API for model fitting and predictions  
**brmspy/types.py** - Type definitions and result dataclasses  
**brmspy/install.py** - R dependency management  
**brmspy/binaries/** - Prebuilt runtime bundle system  
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

## Runtime Binaries System

The runtime system provides prebuilt bundles to skip lengthy R package compilation.

### System Fingerprint

Each runtime is identified by: `{os}-{arch}-r{major}.{minor}`

Examples:
- `linux-x86_64-r4.5`
- `darwin-arm64-r4.5` (macOS Apple Silicon)
- `windows-x86_64-r4.5`

### Components

**Platform Detection (`brmspy/binaries/env.py`):**
- Detect OS, architecture, R version
- Generate system fingerprint
- Check runtime compatibility

**Runtime Building (`brmspy/binaries/build.py`):**
- Bundle CmdStan binaries
- Package R packages (cmdstanr, brms, posterior)
- Include system libraries (Linux)
- Create manifest with metadata

**Runtime Installation (`brmspy/binaries/use.py`):**
- Download from GitHub releases
- Extract and activate runtime
- Configure environment

### Bundle Structure

```
brmspy-runtime-{fingerprint}-{version}.tar.gz
├── manifest.json              # Metadata
├── cmdstan/                   # Compiled CmdStan
├── Rlib                       # R libraries
```

### Usage

```python
import brmspy

# Install prebuilt runtime (2-3 minutes vs 20-30 from source)
brmspy.install_prebuilt(version="0.1.0")

# Build custom runtime locally
from brmspy.binaries.build import build_runtime_bundle
build_runtime_bundle(output_dir="dist/runtime")
```

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
- Tests marked with `@pytest.mark.crossplatform`
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
- Build runtime: python -m brmspy.binaries.build
- Upload tarball
```

**macOS/Windows Build (Native):**
```yaml
- Setup Python 3.12 + R 4.5
- Install dependencies
- Build runtime
- Upload tarball
```

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
├── conftest.py              # Pytest fixtures
├── test_basic.py           # Basic tests
├── test_integration.py     # End-to-end tests
├── test_predictions.py     # Prediction tests
└── test_crossplatform.py   # Cross-platform tests
```

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v --cov=brmspy      # With coverage
pytest -m crossplatform            # Cross-platform only
pytest -n auto                     # Parallel (requires pytest-xdist)
```

### Test Markers

```python
@pytest.mark.crossplatform
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

**Via GitHub Actions:**
1. Go to **Actions** → **runtime-publish**
2. **Run workflow** with version and tag
3. Runtimes published to: `releases/tag/runtime`

**Locally:**
```bash
python -m brmspy.binaries.build --output-dir dist/runtime
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
import brmspy.helpers.rtools as rtools
rtools.install_rtools()
```

### Runtime Incompatibility

```python
# Build local runtime
from brmspy.binaries.build import build_runtime_bundle
build_runtime_bundle(output_dir="custom")

# Or install matching prebuilt
brmspy.install_prebuilt()  # Auto-detects platform
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