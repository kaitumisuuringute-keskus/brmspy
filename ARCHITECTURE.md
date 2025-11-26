# pybrms Architecture Documentation

## Overview

pybrms provides a Pythonic interface to the [brms R package](https://paul-buerkner.github.io/brms/), enabling Python users to leverage brms' powerful Bayesian regression modeling capabilities through Stan.

**Version:** 0.1.0 (Modernized)
**Python Support:** 3.8 - 3.14
**License:** Apache 2.0
**Original Author:** Adam Haber
**Maintainer:** Remi Sebastian Kits
**Repository:** https://github.com/kaitumisuuringute-keskus/pybrms

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Python User Code                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    pybrms Package                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  install_brms(version="latest")                       │  │
│  │  - Version-controlled brms installation               │  │
│  │  - Flexible version specification                     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  fit(formula, data, priors, family, ...)              │  │
│  │  - Main interface for model fitting                   │  │
│  │  - Returns CmdStanPy fit objects                      │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  get_brms_data(dataset_name)                          │  │
│  │  - Helper for brms example datasets                   │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────┬──────────────────────────┬──────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────┐    ┌──────────────────────────┐
│   rpy2 (>=3.5.0)    │    │ CmdStanPy (>=1.2.0)      │
│  - R interface      │    │  - Stan interface        │
│  - Type conversion  │    │  - Model compilation     │
└──────────┬──────────┘    │  - MCMC sampling         │
           │               └──────────┬───────────────┘
           ▼                          ▼
┌─────────────────────┐    ┌──────────────────────────┐
│   R Environment     │    │   CmdStan                │
│  - brms package     │    │  - Stan compiler         │
│  - make_stancode()  │    │  - Sampling engine       │
│  - make_standata()  │    └──────────────────────────┘
└─────────────────────┘
```

## System Components

### 1. Python Layer (pybrms)

**Core Modules:**
- `pybrms/__init__.py`: Package initialization and version info
- `pybrms/pybrms.py`: Main implementation
- `pybrms/_nbdev.py`: nbdev integration metadata

**Key Functions:**

#### `install_brms(version="latest", repo="https://cran.rstudio.com")`
Explicit brms installation with version control.

**Design Rationale:**
- ✅ User controls when and what version to install
- ✅ Prevents silent automatic installations
- ✅ Supports specific versions for reproducibility
- ✅ Clear error messages if brms missing

**Usage:**
```python
import pybrms

# Install latest version
pybrms.install_brms()

# Install specific version
pybrms.install_brms(version="2.23.0")

# Install from custom repository
pybrms.install_brms(version="latest", repo="https://cloud.r-project.org")
```

#### `fit(formula, data, priors, family, ...)`
Main interface for Bayesian model fitting.

**Workflow:**
1. Convert Python data (pandas/dict) → R objects via rpy2
2. Call `brms::make_stancode()` to generate Stan code
3. Call `brms::make_standata()` to prepare data
4. Parse Stan code to ensure correct data types
5. Compile and sample using CmdStanPy
6. Return CmdStanPy fit object

**Return Type:** `cmdstanpy.CmdStanMCMC` (replaces `pystan.StanFit4Model`)

#### `get_brms_data(dataset_name)`
Retrieves example datasets from brms package.

**Examples:**
- `"epilepsy"`: Epilepsy seizure counts
- `"kidney"`: Kidney infection data
- `"inhaler"`: Asthma inhaler data

### 2. Dependency Layer

#### R Interface: rpy2 (>=3.5.0)
- **Purpose**: Bidirectional Python-R communication
- **Key Features**:
  - Type conversion (pandas ↔ R dataframes)
  - R function calling from Python
  - Error handling and exceptions
- **Python 3.8+ Support**: Yes

#### Stan Interface: CmdStanPy (>=1.2.0)
- **Purpose**: Official Python interface to Stan
- **Replaces**: pystan (legacy)
- **Advantages**:
  - Active development and support
  - Better performance
  - More features (pathfinder, variational inference)
  - Official recommendation from Stan team
- **Auto-installation**: CmdStan binary automatically installed on first use

#### Numerical Computing
- **NumPy (>=1.20.0)**: Array operations, data conversion
- **Pandas (>=1.3.0)**: DataFrame handling, data manipulation

### 3. R Layer

#### brms Package
- **Version Support**: Any version (user-controlled)
- **Key Functions Used**:
  - `brms::make_stancode()`: Generate Stan model code
  - `brms::make_standata()`: Prepare data for Stan
  - `brms::bf()`: Create brms formula object
  - `brms::prior_string()`: Define priors
  - `brms::is_brmsprior()`: Validate priors

**Installation:**
Managed by user via `pybrms.install_brms()` function

### 4. Stan Layer

#### CmdStan
- **Automatic Installation**: Handled by CmdStanPy
- **Location**: `~/.cmdstan/` by default
- **Version**: Latest stable (managed by CmdStanPy)

## Data Flow

### Model Fitting Workflow

```
User Input (Python)
  ├─ formula: str = "y ~ x + (1|group)"
  ├─ data: pd.DataFrame
  ├─ priors: list[tuple] = [("normal(0,1)", "b")]
  └─ family: str = "gaussian"
        │
        ▼
pybrms.fit()
  ├─ Step 1: Convert Python → R
  │   └─ pd.DataFrame → R data.frame
  │
  ├─ Step 2: Generate Stan Code (via brms)
  │   └─ brms::make_stancode() → Stan code string
  │
  ├─ Step 3: Prepare Stan Data (via brms)
  │   └─ brms::make_standata() → R list → Python dict
  │
  ├─ Step 4: Type Coercion
  │   └─ Parse Stan code to ensure int/float types match
  │
  ├─ Step 5: Compile & Sample (via CmdStanPy)
  │   ├─ CmdStan compiles Stan code → C++
  │   └─ CmdStan runs MCMC sampling
  │
  └─ Return: CmdStanMCMC object
        │
        ▼
User Analysis (Python)
  ├─ fit.summary()
  ├─ arviz.from_cmdstanpy(fit)
  └─ Visualization with matplotlib/arviz
```

## Type Conversion Pipeline

### Python → R (via rpy2)
```python
pandas.DataFrame → rpy2.robjects.DataFrame
dict → rpy2.robjects.ListVector
numpy.ndarray → rpy2.robjects.vectors
```

### R → Python (via rpy2)
```python
R data.frame → pandas.DataFrame
R list → dict
R vectors → numpy.ndarray
```

### Stan Type Parsing
Special handling required because Stan has strict types:
- `int` variables must be Python `int` (not `float`)
- Arrays parsed from Stan `data{}` block
- Automatic scalar extraction from 1-element arrays

## Configuration Management

### Modern Setup (PEP 517/518/621)

**Primary Configuration:** `pyproject.toml`
- All package metadata
- Dependencies with version constraints
- Optional dependency groups
- Build system configuration
- Tool configurations (pytest, mypy, ruff, black)

**Legacy Configuration:** `setup.py`
- Minimal shim for backward compatibility
- Delegates to pyproject.toml
- Required for some older tools

**nbdev Configuration:** `settings.ini`
- Maintained for nbdev compatibility
- Used by Jupyter notebook workflow
- Synchronized with pyproject.toml version

### Dependency Management Philosophy

**Core Principle:** Minimum version constraints with broad compatibility

```toml
dependencies = [
    "cmdstanpy>=1.2.0",      # Minimum working version
    "numpy>=1.20.0",          # Python 3.8+ compatible
    "pandas>=1.3.0",          # Modern API
    "rpy2>=3.5.0",            # Python 3.8+ support
]
```

**Rationale:**
- ✅ Allows users to use newer versions
- ✅ Doesn't force specific versions
- ✅ Accommodates different environments
- ✅ Reduces dependency conflicts

### Optional Dependencies

**Development:**
```bash
pip install pybrms[dev]  # pytest, black, ruff, mypy
```

**Documentation:**
```bash
pip install pybrms[docs]  # nbdev, jupyter, sphinx
```

**Visualization:**
```bash
pip install pybrms[viz]  # arviz, matplotlib, seaborn
```

**All:**
```bash
pip install pybrms[all]  # Everything
```

## Version Strategy

### Python Version Support
- **Minimum:** Python 3.8 (EOL: October 2024)
- **Maximum:** Python 3.14 (future-proof)
- **Rationale:**
  - 3.6, 3.7 reached EOL
  - Modern type hints and features
  - Better performance
  - Broader ecosystem support

### Semantic Versioning
- **v0.1.0**: Major architecture update (current)
  - Breaking changes from v0.0.3
  - CmdStanPy replaces PyStan
  - Python 3.8+ required
  - Explicit brms installation

### Brms Version Flexibility
- **No pinned version**: User controls brms version
- **Any version supported**: Through rpy2 interface
- **Recommendation**: Use latest stable (2.23.0 as of 2024)

## Breaking Changes from v0.0.3

### 1. Stan Backend
- **Old:** PyStan 2.x (deprecated)
- **New:** CmdStanPy (official interface)
- **Impact:** Different return types and API

### 2. Python Version
- **Old:** Python 3.6+
- **New:** Python 3.8+
- **Impact:** Must upgrade Python version

### 3. brms Installation
- **Old:** Automatic on import (silent)
- **New:** Explicit via `install_brms()`
- **Impact:** Requires manual installation step

### 4. Return Types
- **Old:** `pystan.StanFit4Model`
- **New:** `cmdstanpy.CmdStanMCMC`
- **Impact:** Different methods for accessing results

## Development Workflow

### With nbdev
```bash
# Edit notebooks in root directory
# core.ipynb contains main implementation

# Build library from notebooks
make pybrms

# Build documentation
make docs

# Run tests
make test

# Clean build artifacts
make clean
```

### Standard Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black pybrms/
ruff check pybrms/

# Type checking
mypy pybrms/
```

## Future Architecture Considerations

### Potential Enhancements

1. **Async Support**
   - Async model fitting for long-running samplers
   - Non-blocking R calls

2. **Caching**
   - Cache compiled Stan models
   - Reduce recompilation time

3. **Extended Stan Features**
   - Variational inference via CmdStanPy
   - Pathfinder algorithm
   - Laplace approximation

4. **Better Type Hints**
   - Full type annotations
   - Protocol definitions for return types

5. **Plugin System**
   - Custom prior distributions
   - Custom family functions
   - Model post-processing hooks

## References

- **brms Documentation**: https://paul-buerkner.github.io/brms/
- **CmdStanPy**: https://mc-stan.org/cmdstanpy/
- **rpy2**: https://rpy2.github.io/
- **Stan**: https://mc-stan.org/
- **PEP 517**: https://peps.python.org/pep-0517/
- **PEP 518**: https://peps.python.org/pep-0518/
- **PEP 621**: https://peps.python.org/pep-0621/

## Contributing

See `CONTRIBUTING.md` for development guidelines and architecture decisions.

## License

Apache License 2.0 - See `LICENSE` file for details.