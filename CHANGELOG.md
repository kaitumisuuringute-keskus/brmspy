## 0.1.13 - Enhanced Diagnostics & Type-Safe Summaries
*25.12.04*

### Diagnostics Functions

* **`summary()` Complete Rewrite**: Now returns a `SummaryResult` dataclass instead of a single DataFrame, providing structured access to all model components:
  * Population-level effects (`summary.fixed`)
  * Family-specific parameters (`summary.spec_pars`)
  * Random effects (`summary.random`)
  * Prior specifications (`summary.prior`)
  * Model metadata (formula, chains, draws, diagnostics)
  * Pretty-print support via `__str__()` and `__repr__()` for human-readable output in notebooks
  * Comprehensive docstring with all fields, types, and usage examples

* **`fixef()` Population-Level Effects**: New function for extracting fixed effects with full control:
  * Returns pandas DataFrame with parameter estimates and uncertainty
  * Supports `summary=True` (default) for statistics or `summary=False` for raw posterior samples
  * Configurable credible intervals via `probs` parameter
  * Robust estimates (median/MAD) via `robust=True`
  * Parameter subsetting with `pars` argument
  * Complete docstring with examples for common use cases

* **`ranef()` Group-Level Effects**: New function for extracting random effects as xarray DataArrays:
  * Returns dict mapping grouping factors to multidimensional arrays
  * `summary=True` (default): 3D arrays with dimensions `(group, stat, coef)` containing estimates and intervals
  * `summary=False`: 3D arrays with dimensions `(draw, group, coef)` for full posterior draws
  * Native xarray integration enables easy slicing and MCMC analysis
  * Comprehensive docstring with coord selection examples

* **`posterior_summary()` All Parameters**: New function for comprehensive parameter extraction:
  * Returns pandas DataFrame with all model parameters (fixed, random, auxiliary)
  * More comprehensive than `fixef()` or `ranef()` alone
  * Supports variable selection and regex patterns
  * Configurable credible intervals and robust estimates
  * Complete docstring with filtering examples

* **`prior_summary()` Prior Specifications**: New function for documenting model priors:
  * Returns pandas DataFrame with all prior distributions used in model
  * Option to show all priors (including defaults) or only user-specified
  * Useful for reproducibility and model documentation
  * Complete docstring with comparison examples

* **`loo()` Leave-One-Out Cross-Validation**: New function for model comparison and validation:
  * Computes efficient LOO-CV using Pareto smoothed importance sampling (PSIS)
  * Returns `LooResult` dataclass with LOO metrics (elpd_loo, p_loo, looic) and standard errors
  * Pareto k diagnostics for identifying problematic observations
  * Support for moment matching and refitting for high Pareto k values
  * Pretty-print support via `__repr__()` with formatted metrics and diagnostics
  * Memory-efficient pointwise computation option for large datasets
  * Comprehensive docstring with model comparison examples

* **`loo_compare()` Model Comparison**: New function for comparing multiple models using LOO-CV:
  * Compare 2 or more models using approximate leave-one-out cross-validation
  * Returns `LooCompareResult` dataclass with comparison table ranked by performance
  * Table includes `elpd_diff` (difference from best model) and standard errors
  * Support for custom model names via `model_names` parameter
  * Pretty-print support via `__repr__()` showing formatted comparison table
  * Best model always listed first with elpd_diff = 0
  * Comprehensive docstring with usage examples

* **`validate_newdata()` Data Validation**: New function for validating prediction data:
  * Validates new data against fitted model requirements before making predictions
  * Checks for required variables, factor levels, and grouping structure
  * Returns validated DataFrame ready for use in prediction functions
  * Support for `allow_new_levels` parameter to permit new grouping factor levels
  * Can skip response variable checking with `check_response=False`
  * Validates autocorrelation structures and group-level effects
  * Primarily used internally by prediction methods but available for direct use
  * Comprehensive Python-style docstring with parameter descriptions and examples

### Type System Improvements

* **DataFrame Detection**: `r_to_py()` now correctly detects R DataFrames and preserves:
  * Row indexes (rownames from R)
  * Column names
  * Proper type conversion for all data types

* **LooResult Dataclass**: Added `__repr__()` for formatted LOO-CV output:
  * Displays ELPD LOO, p_loo, and LOOIC with standard errors
  * Summary of Pareto k diagnostic categories
  * Human-readable format for notebooks and interactive use

### Testing

* Added 14 comprehensive tests for diagnostics functions:
  * `summary()`: structure validation, component access, pretty printing
  * `fixef()`: summary statistics and parameter extraction
  * `ranef()`: both summary mode and raw posterior draws mode
  * `posterior_summary()`: all parameters extraction
  * `prior_summary()`: prior specifications with custom priors
  * `loo()`: basic LOO-CV computation and Pareto k diagnostics
  * `loo_compare()`: model comparison with default and custom model names
  * `validate_newdata()`: validation with valid data and error handling for invalid data
* All tests optimized with reduced iterations (`iter=100, warmup=50`) for faster CI execution

### Generic Function Access

* **`call()` Universal Function Wrapper**: New generic function for calling any brms or R function:
  * Call brms functions by name without dedicated Python wrappers
  * Automatic bidirectional type conversion (Python ↔ R)
  * Tries `brms::function_name` first, falls back to base R
  * Useful for accessing newer brms features or utility functions
  * Complete docstring with examples for various use cases

* **`sanitised_name()` Helper Function**: Sanitizes function names for safe R execution:
  * Converts Python-style names to valid R identifiers
  * Preserves namespace separators (e.g., `brms::loo`)
  * Handles invalid characters and numeric prefixes
  * Complete docstring with sanitization examples

### API

* Exported `fixef`, `ranef`, `posterior_summary`, `prior_summary`, `loo`, `loo_compare`, `validate_newdata`, and `call` from `brmspy` module
* Added `LooResult` and `LooCompareResult` to public types for model comparison workflows


## 0.1.12 - RDS loading/saving, families functions, default priors functions
*25.12.03*

### New Features

* Added `save_rds()` for saving `brmsfit` or generic R objects.
* Added `load_rds_fit()` for loading saved `brmsfit` objects and returning a `FitResult` with attached `InferenceData`.
* Added `load_rds_raw()` for loading arbitrary R objects from RDS files.
* Added `fit` alias `brm`.

### Families

* Added `brmspy.families` module with Python wrappers for `brmsfamily()` and `family()`.
* Implemented keyword-argument wrappers for the following families:
  `student`, `bernoulli`, `beta_binomial`, `negbinomial`, `geometric`,
  `lognormal`, `shifted_lognormal`, `skew_normal`, `exponential`, `weibull`,
  `frechet`, `gen_extreme_value`, `exgaussian`, `wiener`,
  `Beta`, `dirichlet`, `logistic_normal`, `von_mises`, `asym_laplace`, `cox`,
  `hurdle_poisson`, `hurdle_negbinomial`, `hurdle_gamma`, `hurdle_lognormal`,
  `hurdle_cumulative`, `zero_inflated_beta`, `zero_one_inflated_beta`,
  `zero_inflated_poisson`, `zero_inflated_negbinomial`,
  `zero_inflated_binomial`, `zero_inflated_beta_binomial`,
  `categorical`, `multinomial`, `cumulative`, `sratio`, `cratio`, `acat`.

### Priors

* Added `default_prior()` for retrieving default priors for a model formula and dataset.
* Added `get_prior()` for inspecting prior structure before fitting.

### API Organization

* Reorganized brms wrappers into modular files under `brmspy/brms_functions/`
  (`brm`, `diagnosis`, `families`, `formula`, `io`, `prediction`, `prior`, `stan`).

### Internal / Typing

* Added `RListVectorExtension` protocol for return types that wrap R list-like structures.
  Enables automatic extraction of underlying R objects in `py_to_r` and `kwargs_r`.


## 0.1.11 - Persistent Runtimes, Modern Packaging, and Stability
*25.12.01*

*   **Persistent Runtimes**: If a prebuilt runtime is activated, the path is now saved to `{HOME}/.brmspy/config.json` and automatically loaded on the next library import.
*   **Configurable Logging**: Replaced scattered print statements with a centralized logger.
*   **Optimized Activation**: Made `activate_runtime` aggressive unloading conditional, resulting in slightly faster runtime activation.



## 0.1.10 - Windows stability & CI improvements
*25.12.01*

**Windows Support**

*   Implemented aggressive R package unloading (detach, unloadNamespace, DLL unload) prior to installation to prevent file locking errors.
*   Refined RTools detection logic; relaxed system `g++` version requirements when valid RTools installations are detected.
*   Changed `install_rtools` default to `False` in `install_brms()` to prevent unintended system PATH modifications.
*   Fixed PowerShell command syntax generation for internal system calls.
*   Windows prebuilt binaries currently requires R4.5, more versions are coming.

**Build and CI**

*   Expanded continuous integration test matrix to cover Windows, macOS, and Ubuntu on Python 3.12.
*   Optimized GitHub Actions caching strategy for R libraries and CmdStan, significantly reducing Windows workflow runtime.
*   Fixed artifact pruning logic in runtime builder workflows to prevent test breakages.

**Bug Fixes**

*   Ensured `jsonlite` dependency is explicitly resolved during build manifest generation.
*   Fixed workflow path referencing and quoting issues in installation scripts.



## 0.1.9 - Prebuilt runtimes, windows toolchain
*25.11.30*

**New Features**

*   **Prebuilt Runtimes**: Introduced `brmspy.binaries` subpackage to manage precompiled R environments containing `brms` and `cmdstanr` (installs up to 50x faster than default).
*   **Fast Installation**: Added `use_prebuilt_binaries=True` argument to `install_brms()`, reducing setup time by bypassing source compilation.
*   **Windows Toolchain**: Implemented automatic detection and installation of Rtools (MinGW-w64) within `install_brms()` to support compilation on Windows.

**Enhancements**

*   **Linux Binaries**: Updated Linux installation logic to prioritize Posit Package Manager (P3M) binary repositories based on OS codename detection.
*    **Extended documentation**: Added docstring documentation to all public and internal functions.

**Infrastructure**

*   **Docker**: Added `.runtime_builder` Dockerfiles for creating reproducible Linux runtime environments.



## 0.1.8 - Rstan support, version pinning, and cross-platform fixes
*25.11.29*

**Features**

*   **Rstan Backend Support:** Added `rstan` as an alternative backend. install_brms now takes `install_rstan` param and `fit()` now seamlessly accepts `backend="rstan"` (defaults to "cmdstanr").
*   **Exact Version Installation:** `install_brms()` now supports pinning specific R package versions (e.g., `version="2.21.0"`) using `remotes` for all R dependencies.

**Installation & Platform Support**

*   **Windows Toolchain:** Implemented automatic Rtools detection and setup within `install_brms()`, resolving compiler issues on Windows.
*   **macOS/Windows Binaries:** Fixed installation failures for dependencies (like `data.table`) by defaulting to `type="both"` instead of forcing source compilation.

**Infrastructure**

*   Added dedicated cross-platform CI workflow (Windows, macOS, Ubuntu) to ensure installation stability across operating systems.



## 0.1.7 - Automatic r imports after installation and import failure warning
*25.11.29*

* Fixed an issue where the library could refuse import in case of missing R deps
* R libraries are now automatically imported after installation

## 0.1.6 - Fix segfaults, stabilize R imports, faster cold starts
*25.11.29*

### Core stability

* **Fixed a rare but catastrophic segfault** that could occur when `fit()` was invoked inside a `tqdm` loop or any repeated call context with dynamic stdouts.

  * Root cause: calling `rpackages.importr("cmdstanr")` / `importr("brms")` *inside* functions caused rpy2 to repeatedly rebuild R package proxies, which in some environments led to segfaults and the Python process or even the IDE being killed.
  * **Resolution:** All R-side imports (`brms`, `cmdstanr`, `posterior`, etc.) are now performed **once at module import time**, *never inside functions*. This completely eliminates the segfault and removes hidden global-state churn.

**Performance & architecture**

* Because heavy rpy2 `importr(...)` calls are no longer done on-demand during `fit()`, repeated model fits are now **measurably faster** (no repeated R namespace setup).

**API invariants & guarantees**

* `fit(sample=False)` is now guaranteed safe to use in loops, notebooks, and benchmarking environments.
* Repeated calls to `fit()` no longer risk triggering implicit R package reloads, leading to **consistent runtime semantics**.

**Testing**

* Added a dedicated regression test:
  `test_fit_tqdm_segfault()` — ensures that calling `fit()` inside a `tqdm` loop is stable and does not crash the interpreter.

  * This test reproduces the exact minimal condition under which the old bug appeared and will prevent regressions long-term.

**Misc**

* Minor internal cleanups and defensive guards around lazy globals (`_brms`, etc.) to ensure they initialize exactly once and never re-import.
* Slightly reduced memory churn on repeated calls by eliminating redundant converter/namespace setup.



## 0.1.5 – Better priors, kwargs parsing, formula helper, typed ArviZ
*25.11.28*

**API & types**

* Bumped version to `0.1.5` and exported more typed return types from the top-level package.
* Added a public `formula()` helper and exported it from `brmspy` so you can pass kwargs to it, prebuild / reuse model formulas instead of always passing strings straight into `fit()`.
* Introduced typed ArviZ aliases for the different `InferenceData` shapes returned by various functions (`IDFit`, `IDPredict`, `IDLinpred`, `IDLogLik`, `IDEpred`) and re-exported them in `__all__` for IDE/type-checker friendliness. 
* Tightened / clarified the main return types: `FitResult`, `PosteriorEpredResult`, `PosteriorPredictResult`, `PosteriorLinpredResult`, `LogLikResult`, `GenericResult` are now consistently exported and documented as the official public surface.

**Priors**

* Promoted the `prior()` helper to the recommended way of specifying priors; examples and docs now use `prior(...)` instead of raw tuples.
* Switched the main Quick Start example to more sensible defaults for the epilepsy model:

  * `prior("normal(0, 1)", "b")`
  * `prior("exponential(1)", "sd", group="patient")`
  * `prior("student_t(3, 0, 2.5)", "Intercept")`
* Added / refined internal prior-building logic (commit: “proper prior building”) so Python-side specs map more reliably onto `brms::set_prior()` and accept richer combinations of `class_`, `coef`, `group`, `dpar`, etc. 
* Added tests around `prior()` and how priors are passed into `fit()`. (commit: “tests for priors, get_stancode, summary, fit without sampling, poster…”)

**`fit()` and kwargs**

* Improved `fit()` kwargs parsing so that Python kwargs are forwarded more robustly to `brms`/`cmdstanr`
  * Makes it safer to pass through extra control arguments without silently dropping or misrouting them.

**Tooling & infra**

* Expanded and tightened test coverage in `tests/test_basic.py` and `tests/test_integration.py`, including coverage for priors, `get_stancode`/`make_stancode`, `summary`, and “fit without sampling” paths. 
