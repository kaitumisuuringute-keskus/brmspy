## 0.2.1 - Stability hotfix
*25.12.07*

* Try to enforce rpy2 RPY2_CFFI_MODE ABI mode on import with warnings if not possible. API/BOTH can cause instability on linux and macos (Issue: #45)
* Added R_HOME and LD_LIBRARY_PATH to github workflows (required on most environments in ABI mode)
* install_rstan in `install_brms` now defaults to False.
* The environment now does its best attempts to detect invalid R setups and log them


## 0.2.0 - Runtime Refactor & Formula DSL
*25.12.07*

### Breaking Changes

*   **Removed Diagnostics**: Removed `loo`, `loo_compare`, and `add_criterion` due to frequent segfaults in embedded R mode. Users should rely on `arviz.loo` and `arviz.compare` using the `idata` property of the fit result.
*   **Installation API**: Renamed `use_prebuilt_binaries` argument to `use_prebuilt` in `install_brms()`.
*   Installation API now consists of: `install_brms`, `install_runtime`, `deactivate_runtime`, `activate_runtime`, `find_local_runtime`, `get_active_runtime`, `get_brms_version`
*   **Deprecations**: Renamed `fit` to `brm` and `formula` to `bf`. Previous names are still exported as aliases, but might be removed in a future version.

### New Features

*   **Formula DSL**: Implemented `bf`, `lf`, `nlf`, `acformula`, `set_rescor`, `set_mecor`, and `set_nl`. These objects support additive syntax (e.g., `bf(...) + set_rescor(True) + gaussian()`) mirroring native brms behavior.
*   **Generic Data Loader**: Added `get_data()` to load datasets from any installed R package, complementing `get_brms_data()`.
*   **Runtime Status**: Added `brmspy.runtime.status()` to programmatically inspect the current R environment, toolchain compatibility, and active runtime configuration.
*   **Families now in package root**: Families can now be imported from package root, e.g `from brmspy import gaussian`

### Runtime & Installation

*   **Core Refactor**: Completely re-architected `brmspy.runtime` into strict layers (`_config`, `_r_env`, `_platform`, `_install`, etc) to eliminate side effects during import and prevent circular dependencies.
*   **Atomic Activation**: `activate_runtime()` now validates manifest integrity and system fingerprints before mutating the R environment, ensuring atomic success or rollback.
*   **Auto-Persistence**: The last successfully activated runtime is automatically restored on module import via `runtime._autoload`, creating persistent sessions across restarts.
*   **Windows Toolchain**: Modularized RTools detection logic to accurately map R versions to RTools versions (4.0–4.5) and handle path updates safely.

### Documentation & Infrastructure

*   **MkDocs Migration**: Ported all documentation to MkDocs with the Material theme for better navigability and API references.
*   **Rendered notebooks**: Added more notebook examples that are now rendered fully with links to running each in Google Colab.
*   **ArViz diagnostics examples**: can now be found under API reference
*   **Test coverage**: Test coverage for brms functions is now at 88% and for R environment and package management at 68%




## 0.1.13 - Enhanced Diagnostics & Type-Safe Summaries
*25.12.04*

### Diagnostics

*   **`summary()` Rewrite**: Returns `SummaryResult` dataclass with structured access to `fixed`, `spec_pars`, `random`, `prior`, and model metadata. Includes pretty-print support.
*   **`fixef()`**: Extract population-level effects as DataFrame. Supports `summary`, `robust`, `probs`, and `pars` arguments.
*   **`ranef()`**: Extract group-level effects as xarray DataArrays. Returns dict mapping grouping factors to arrays with configurable summary/raw modes.
*   **`posterior_summary()`**: Extract all model parameters (fixed, random, auxiliary) as DataFrame. Supports variable selection and regex patterns.
*   **`prior_summary()`**: Return DataFrame of prior specifications. Option to show all priors or only user-specified.
*   **`loo()`**: Compute LOO-CV using PSIS. Returns `LooResult` with elpd_loo, p_loo, looic, and Pareto k diagnostics.
*   **`loo_compare()`**: Compare multiple models via LOO-CV. Returns `LooCompareResult` ranked by performance with `elpd_diff` and standard errors.
*   **`validate_newdata()`**: Validate prediction data against fitted model requirements. Checks variables, factor levels, and grouping structure.

### Type System

*   **DataFrame Detection**: `r_to_py()` now correctly preserves row indexes, column names, and proper type conversion from R DataFrames.
*   **`LooResult`/`LooCompareResult`**: Added `__repr__()` for formatted notebook output.

### Generic Function Access

*   **`call()`**: Universal wrapper for calling any brms or R function by name with automatic type conversion.
*   **`sanitised_name()`**: Helper to convert Python-style names to valid R identifiers.

### Testing

*   Added 14 tests covering all new diagnostics functions.
*   Optimized test iterations (`iter=100, warmup=50`) for faster CI.


## 0.1.12 - RDS I/O, Families Module, Default Priors
*25.12.03*

### New Features

*   **`save_rds()`**: Save `brmsfit` or generic R objects to RDS files.
*   **`load_rds_fit()`**: Load saved `brmsfit` objects, returning `FitResult` with attached `InferenceData`.
*   **`load_rds_raw()`**: Load arbitrary R objects from RDS files.
*   **`brm` Alias**: Added `brm` as alias for `fit`.

### Families

*   Added `brmspy.families` module with `brmsfamily()` and `family()` wrappers.
*   Implemented keyword-argument wrappers for 40+ families: `student`, `bernoulli`, `beta_binomial`, `negbinomial`, `geometric`, `lognormal`, `shifted_lognormal`, `skew_normal`, `exponential`, `weibull`, `frechet`, `gen_extreme_value`, `exgaussian`, `wiener`, `Beta`, `dirichlet`, `logistic_normal`, `von_mises`, `asym_laplace`, `cox`, `hurdle_*`, `zero_inflated_*`, `categorical`, `multinomial`, `cumulative`, `sratio`, `cratio`, `acat`.

### Priors

*   **`default_prior()`**: Retrieve default priors for a model formula and dataset.
*   **`get_prior()`**: Inspect prior structure before fitting.

### Internal

*   Reorganized brms wrappers into modular files under `brmspy/brms_functions/`.
*   Added `RListVectorExtension` protocol for automatic R list extraction in type conversion.


## 0.1.11 - Persistent Runtimes & Logging
*25.12.01*

### New Features

*   **Persistent Runtimes**: Activated runtime path saved to `~/.brmspy/config.json` and auto-loaded on import.
*   **Configurable Logging**: Replaced print statements with centralized logger.
*   **Optimized Activation**: Made aggressive unloading conditional for faster runtime activation.


## 0.1.10 - Windows Stability & CI Improvements
*25.12.01*

### Windows Support

*   Implemented aggressive R package unloading (detach, unloadNamespace, DLL unload) to prevent file locking errors.
*   Refined RTools detection; relaxed `g++` version requirements when valid RTools is detected.
*   Changed `install_rtools` default to `False` in `install_brms()` to prevent unintended PATH modifications.
*   Fixed PowerShell command syntax generation.
*   Windows prebuilt binaries currently require R4.5.

### Build & CI

*   Expanded CI matrix: Windows, macOS, Ubuntu on Python 3.12.
*   Optimized GitHub Actions caching for R libraries and CmdStan.
*   Fixed artifact pruning logic in runtime builder workflows.

### Bug Fixes

*   Ensured `jsonlite` dependency is explicitly resolved during manifest generation.
*   Fixed workflow path referencing and quoting issues.


## 0.1.9 - Prebuilt Runtimes & Windows Toolchain
*25.11.30*

### New Features

*   **Prebuilt Runtimes**: Added `brmspy.binaries` subpackage for precompiled R environments with `brms` and `cmdstanr` (up to 50x faster install).
*   **Fast Installation**: Added `use_prebuilt_binaries=True` argument to `install_brms()`.
*   **Windows Toolchain**: Automatic Rtools (MinGW-w64) detection and installation in `install_brms()`.

### Enhancements

*   **Linux Binaries**: Prioritize Posit Package Manager (P3M) binary repositories based on OS codename.
*   **Documentation**: Added docstrings to all public and internal functions.

### Infrastructure

*   Added `.runtime_builder` Dockerfiles for reproducible Linux runtime environments.


## 0.1.8 - RStan Support & Version Pinning
*25.11.29*

### New Features

*   **RStan Backend**: Added `rstan` as alternative backend. `install_brms()` accepts `install_rstan` param; `fit()` accepts `backend="rstan"`.
*   **Version Pinning**: `install_brms()` supports pinning specific R package versions (e.g., `version="2.21.0"`) via `remotes`.

### Platform Support

*   **Windows Toolchain**: Automatic Rtools detection and setup in `install_brms()`.
*   **macOS/Windows Binaries**: Fixed installation failures by defaulting to `type="both"` instead of forcing source compilation.

### Infrastructure

*   Added cross-platform CI workflow (Windows, macOS, Ubuntu).


## 0.1.7 - Import Fixes
*25.11.29*

*   Fixed library refusing import when R dependencies are missing.
*   R libraries now automatically imported after installation.


## 0.1.6 - Segfault Fix & Stability
*25.11.29*

### Core Stability

*   **Fixed segfault** occurring when `fit()` was called inside `tqdm` loops or repeated call contexts.
*   All R imports (`brms`, `cmdstanr`, `posterior`) now performed once at module import, never inside functions.

### Performance

*   Repeated model fits now faster due to eliminated R namespace reloads.
*   Reduced memory churn by removing redundant converter/namespace setup.

### Testing

*   Added `test_fit_tqdm_segfault()` regression test.


## 0.1.5 - Priors, Formula Helper, Typed ArviZ
*25.11.28*

### API & Types

*   **`formula()`**: Added helper for building reusable model formulas with kwargs support.
*   **Typed ArviZ Aliases**: Added `IDFit`, `IDPredict`, `IDLinpred`, `IDLogLik`, `IDEpred` for different `InferenceData` shapes.
*   **Exported Types**: `FitResult`, `PosteriorEpredResult`, `PosteriorPredictResult`, `PosteriorLinpredResult`, `LogLikResult`, `GenericResult` now in public API.

### Priors

*   **`prior()` Helper**: Now recommended way to specify priors instead of raw tuples.
*   Improved internal prior-building logic for better mapping to `brms::set_prior()`.
*   Supports `class_`, `coef`, `group`, `dpar` combinations.

### Internal

*   Improved `fit()` kwargs parsing for more robust forwarding to `brms`/`cmdstanr`.
*   Expanded test coverage for priors, `get_stancode`, `summary`, and fit-without-sampling paths.