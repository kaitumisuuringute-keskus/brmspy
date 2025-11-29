## 0.1.9 - 



## 0.1.8 - Rstan support, version pinning, and cross-platform fixes

**Features**

*   **Rstan Backend Support:** Added `rstan` as an alternative backend. install_brms now takes `install_rstan` param and `fit()` now seamlessly accepts `backend="rstan"` (defaults to "cmdstanr").
*   **Exact Version Installation:** `install_brms()` now supports pinning specific R package versions (e.g., `version="2.21.0"`) using `remotes` for all R dependencies.

**Installation & Platform Support**

*   **Windows Toolchain:** Implemented automatic Rtools detection and setup within `install_brms()`, resolving compiler issues on Windows.
*   **macOS/Windows Binaries:** Fixed installation failures for dependencies (like `data.table`) by defaulting to `type="both"` instead of forcing source compilation.

**Infrastructure**

*   Added dedicated cross-platform CI workflow (Windows, macOS, Ubuntu) to ensure installation stability across operating systems.


## 0.1.7 - Automatic r imports after installation and import failure warning

* Fixed an issue where the library could refuse import in case of missing R deps
* R libraries are now automatically imported after installation

## 0.1.6 - Fix segfaults, stabilize R imports, faster cold starts

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
