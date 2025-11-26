# Migration Guide: v0.0.3 → v0.1.0

This guide helps you migrate from pybrms v0.0.3 to the modernized v0.1.0 release.

## Overview of Changes

pybrms v0.1.0 represents a major modernization of the package with several breaking changes:

| Aspect | v0.0.3 (Old) | v0.1.0 (New) |
|--------|--------------|--------------|
| **Python Version** | 3.6+ | 3.8+ |
| **Stan Backend** | PyStan 2.x | CmdStanPy |
| **brms Installation** | Automatic (silent) | Explicit via `install_brms()` |
| **Configuration** | setup.py only | pyproject.toml (PEP 621) |
| **Return Type** | `pystan.StanFit4Model` | `cmdstanpy.CmdStanMCMC` |
| **Repository** | github.com/adamhaber/pybrms | github.com/kaitumisuuringute-keskus/pybrms |

## Breaking Changes

### 1. Python Version Requirement

**Old:** Python 3.6+
**New:** Python 3.8+

**Why:** Python 3.6 and 3.7 reached end-of-life. Modern libraries require Python 3.8+.

**Action Required:**
```bash
# Check your Python version
python --version

# If < 3.8, upgrade Python
# Then reinstall pybrms
pip install --upgrade pybrms
```

### 2. Stan Backend Change

**Old:** PyStan 2.x (deprecated)
**New:** CmdStanPy (official interface)

**Why:** PyStan 2.x is in maintenance mode. CmdStanPy is actively developed and officially recommended by the Stan team.

**Impact:**
- Different return types
- Different methods for accessing results
- Better performance and features

**Example Changes:**

```python
# OLD (v0.0.3) - PyStan
import pybrms
model = pybrms.fit(formula, data, family="poisson")
print(model)  # PyStan StanFit4Model object
samples = model.extract()['b']  # Extract parameters

# NEW (v0.1.0) - CmdStanPy
import pybrms
pybrms.install_brms()  # First time only
model = pybrms.fit(formula, data, family="poisson")
print(model.summary())  # CmdStanPy summary
samples = model.draws()[:, :, 'b']  # Extract parameters
```

**Parameter Access:**

```python
# OLD - PyStan
samples = fit.extract()
beta = samples['b']  # Simple dict access
intercept = samples['Intercept']

# NEW - CmdStanPy
summary = fit.summary()  # Pandas DataFrame
beta = fit.draws()[:, :, 'b']  # NumPy array indexing
# Or use arviz for easier access:
import arviz as az
idata = az.from_cmdstanpy(fit)
beta = idata.posterior['b']  # xarray DataArray
```

### 3. Explicit brms Installation

**Old:** brms automatically installed on first import (silent)
**New:** Explicit installation required via `install_brms()`

**Why:** 
- Users should control when and what version to install
- Prevents surprise downloads during import
- Enables version pinning for reproducibility

**Migration:**

```python
# OLD - Automatic (v0.0.3)
import pybrms
# brms automatically installed if missing
model = pybrms.fit(...)

# NEW - Explicit (v0.1.0)
import pybrms

# First time setup - install brms
pybrms.install_brms()  # Latest version
# OR
pybrms.install_brms(version="2.23.0")  # Specific version

# Then use normally
model = pybrms.fit(...)
```

**One-Time Setup:**
After installing pybrms, run once per environment:
```python
import pybrms
pybrms.install_brms()  # Installs latest brms in R
```

### 4. Dependency Updates

**Updated Packages:**

| Package | Old | New | Notes |
|---------|-----|-----|-------|
| pystan | >=2.17 | **REMOVED** | Replaced by cmdstanpy |
| cmdstanpy | N/A | >=1.2.0 | **NEW** official interface |
| numpy | >=1.16 | >=1.20 | Python 3.8+ compatible |
| pandas | >=0.24 | >=1.3 | Modern API |
| rpy2 | >=3.1 | >=3.5 | Python 3.8+ support |

**Action Required:**
```bash
# Recommended: Fresh install in new environment
pip uninstall pybrms pystan
pip install pybrms  # Installs v0.1.0 with new dependencies
```

### 5. Result Visualization

**Old:** Compatible with PyStan-based tools
**New:** Compatible with CmdStanPy-based tools

```python
# OLD - PyStan visualization
import pybrms
fit = pybrms.fit(...)
fit.plot()  # PyStan built-in plotting

# NEW - Use arviz for visualization
import pybrms
import arviz as az
import matplotlib.pyplot as plt

fit = pybrms.fit(...)
idata = az.from_cmdstanpy(fit)
az.plot_trace(idata)
az.plot_posterior(idata, var_names=['b', 'Intercept'])
plt.show()
```

**Install visualization tools:**
```bash
pip install "pybrms[viz]"  # Includes arviz, matplotlib, seaborn
```

## Step-by-Step Migration

### Step 1: Check Python Version
```bash
python --version
# Should be >= 3.8
```

If older than 3.8, upgrade Python first.

### Step 2: Create New Virtual Environment (Recommended)
```bash
# Recommended to avoid conflicts
python -m venv pybrms-env
source pybrms-env/bin/activate  # Linux/Mac
# pybrms-env\Scripts\activate  # Windows

pip install pybrms
```

### Step 3: Install brms (One-Time Setup)
```python
import pybrms
pybrms.install_brms()  # Latest version
# Wait for installation to complete
```

### Step 4: Update Your Code

**Basic Usage (minimal changes):**
```python
# Before (v0.0.3)
import pybrms
epilepsy = pybrms.get_brms_data("epilepsy")
model = pybrms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson"
)

# After (v0.1.0) - Same interface!
import pybrms
pybrms.install_brms()  # First time only
epilepsy = pybrms.get_brms_data("epilepsy")
model = pybrms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson"
)
```

**Accessing Results (changed):**
```python
# Before - PyStan methods
samples = model.extract()
beta = samples['b']
model.plot()

# After - CmdStanPy methods
summary = model.summary()  # Pandas DataFrame
print(summary)

# For parameter access, use arviz
import arviz as az
idata = az.from_cmdstanpy(model)
az.plot_posterior(idata, var_names=['b'])
```

### Step 5: Update Visualization Code

```python
# Before (v0.0.3)
import pybrms
model = pybrms.fit(...)
model.plot()  # PyStan plotting

# After (v0.1.0)
import pybrms
import arviz as az
import matplotlib.pyplot as plt

model = pybrms.fit(...)
idata = az.from_cmdstanpy(model)

# Rich visualizations with arviz
az.plot_trace(idata)
az.plot_posterior(idata)
az.plot_pair(idata, var_names=['b'])
az.plot_forest(idata)
plt.show()
```

## Common Migration Issues

### Issue 1: "No module named 'pystan'"

**Cause:** Code still imports pystan directly
**Solution:** Remove direct pystan imports, use pybrms interface only

```python
# DON'T
import pystan  # Not needed anymore

# DO
import pybrms
# pybrms handles Stan internally via CmdStanPy
```

### Issue 2: "AttributeError: 'CmdStanMCMC' object has no attribute 'extract'"

**Cause:** Using PyStan methods on CmdStanPy object
**Solution:** Update to CmdStanPy methods or use arviz

```python
# OLD - PyStan
samples = fit.extract()

# NEW - CmdStanPy + arviz
import arviz as az
idata = az.from_cmdstanpy(fit)
samples = idata.posterior
```

### Issue 3: "brms not found" error

**Cause:** brms not explicitly installed
**Solution:** Run install_brms()

```python
import pybrms
pybrms.install_brms()  # Run once per environment
```

### Issue 4: Python version conflicts

**Cause:** Python < 3.8
**Solution:** Upgrade Python

```bash
# Check version
python --version

# Upgrade Python (method depends on OS)
# Then create fresh environment
python3.8 -m venv myenv
source myenv/bin/activate
pip install pybrms
```

## Feature Parity Table

| Feature | v0.0.3 | v0.1.0 | Notes |
|---------|--------|--------|-------|
| Model Fitting | ✅ | ✅ | Same interface |
| Custom Priors | ✅ | ✅ | Same interface |
| Family Functions | ✅ | ✅ | Same interface |
| Sample Control | ✅ | ✅ | Same interface |
| Extract Samples | ✅ | ✅ | Different method |
| Built-in Plotting | ✅ | ⚠️ | Use arviz instead |
| Diagnostics | ✅ | ✅ | Better with arviz |
| Model Comparison | ⚠️ | ✅ | Better with arviz |
| Parallel Chains | ✅ | ✅ | Better performance |

## Benefits of Upgrading

### Performance
- ✅ Faster compilation (CmdStanPy)
- ✅ Better parallel execution
- ✅ More efficient memory usage

### Features
- ✅ Access to latest Stan features
- ✅ Better diagnostics via arviz
- ✅ More visualization options
- ✅ Pathfinder algorithm support
- ✅ Variational inference support

### Maintainability
- ✅ Active development (CmdStanPy)
- ✅ Better error messages
- ✅ Modern Python standards
- ✅ Better documentation

### Version Control
- ✅ Explicit brms version management
- ✅ Reproducible environments
- ✅ No surprise installations

## Rollback (If Needed)

If you need to temporarily use the old version:

```bash
# Install old version
pip install pybrms==0.0.3

# Note: Requires Python 3.6-3.7
# Old version uses PyStan 2.x
```

## Getting Help

**Issues with migration?**
- GitHub Issues: https://github.com/kaitumisuuringute-keskus/pybrms/issues
- Original Repo: https://github.com/adamhaber/pybrms

**Documentation:**
- CmdStanPy: https://mc-stan.org/cmdstanpy/
- brms: https://paul-buerkner.github.io/brms/
- arviz: https://arviz-devs.github.io/arviz/

**Examples:**
See updated examples in README.md and documentation.

## Timeline

| Version | Status | Python | Stan Backend |
|---------|--------|--------|--------------|
| 0.0.3 | Legacy | 3.6+ | PyStan 2.x |
| 0.1.0 | Current | 3.8+ | CmdStanPy |
| Future | Planned | 3.9+ | CmdStanPy |

## Contributors

**Original Author:** Adam Haber
**Maintainer (v0.1.0):** Remi Sebastian Kits

Special thanks to the Stan team for CmdStanPy and the brms team for the amazing R package.

## License

Apache License 2.0 - See LICENSE file for details.