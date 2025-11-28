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
