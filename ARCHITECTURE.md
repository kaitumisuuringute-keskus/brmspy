# brmspy Architecture

Technical design, implementation choices, and compatibility details for brmspy v0.1.0.

## System Overview

brmspy is a Python wrapper around R's brms package that provides:
1. Pythonic interface with proper parameter names
2. Native Python Bayesian ecosystem integration (arviz)
3. Optional R brmsfit access for advanced users

**Stack:**
```
Python (3.8-3.14)
    ↓
rpy2 (Python-R bridge)
    ↓
brms (R package)
    ↓
cmdstanr (R's Stan interface)
    ↓
CmdStan (Stan compiler)
```

## Core Design Decisions

### 1. Return Type Strategy

**Problem:** Original implementation returned R objects (brmsfit) with generic parameter names (`b_dim_0`, `sd_1_dim_0`).

**Solution:** Configurable `return_type` parameter with three modes:

| Mode | Return Type | Use Case | Users |
|------|-------------|----------|-------|
| `"idata"` (default) | `arviz.InferenceData` | Python analysis | Most users |
| `"brmsfit"` | R `brmsfit` object | R methods | Advanced |
| `"both"` | `BrmsFitResult` wrapper | Maximum flexibility | Power users |

**Implementation:**
- [`BrmsFitResult`](brmspy/brmspy.py:435-470) - Wrapper class with `.idata` and `.brmsfit` attributes
- [`_brmsfit_to_idata()`](brmspy/brmspy.py:503-567) - Conversion via posterior R package
- [`fit()`](brmspy/brmspy.py:569-789) - Main interface with return type handling

### 2. Parameter Naming Fix

**Problem:** Using CmdStanPy directly bypassed brms' parameter renaming, causing generic names.

**Solution:** Use brms' native `brm()` function with cmdstanr backend instead of:
1. Generating Stan code with `make_stancode()`
2. Generating data with `make_standata()`
3. Running CmdStanPy directly

**Why This Works:**
- brms' `brm()` handles parameter renaming internally
- cmdstanr backend provides proper Stan integration
- Full brms ecosystem benefits (diagnostics, summaries, etc.)

**Trade-off:** Less control over Stan execution, but proper names are critical.

### 3. Type Coercion Pipeline

**Problem:** R→Python→Stan type conversions cause issues:
- R integers → pandas float64 (due to NA handling)
- Stan requires strict int vs float types
- New Stan array syntax: `array[N] int Y` vs old `int Y[N]`

**Solution:** [`_coerce_types()`](brmspy/brmspy.py:361-433) function:

```python
# Parse Stan data block to identify required types
# Handle both old and new Stan array syntax
# Coerce Python types accordingly
```

**Implementation Details:**
1. Extract data block from Stan code via regex
2. Parse variable declarations (handles both syntaxes)
3. Convert 1-size arrays to scalars FIRST
4. Apply int/float coercion based on Stan declarations

**Key Fix (Lines 393-412):**
```python
# New syntax: array[N] int Y
if tokens[0] == 'array' and len(tokens) >= 3:
    var_types.append(tokens[1])  # Type is second token
    var_names.append(tokens[-1])
# Old syntax: int Y[N]
elif len(tokens) >= 2:
    var_types.append(tokens[0])  # Type is first token
    var_names.append(tokens[-1])
```

## Data Flow

### Model Fitting Pipeline

```
1. User calls fit()
   ↓
2. Convert Python data → R (pandas2ri, numpy2ri)
   ↓
3. Call brms::brm() with cmdstanr backend
   ↓
4. brms generates Stan code + data internally
   ↓
5. cmdstanr compiles and samples
   ↓
6. brms returns brmsfit with proper names
   ↓
7. Convert based on return_type:
   - "idata": posterior::as_draws_df() → arviz.InferenceData
   - "brmsfit": return R object directly
   - "both": wrap in BrmsFitResult
```

### Type Conversion Flow

```
R data (brms::make_standata)
   ↓
rpy2 converters (pandas2ri, numpy2ri)
   ↓
Python/pandas/numpy objects
   ↓
_coerce_types() based on Stan data block
   ↓
Properly typed data for Stan
```

## Component Details

### Core Functions

**[`install_brms()`](brmspy/brmspy.py:22-162)**
- Installs cmdstanr R package from r-universe
- Installs CmdStan compiler via cmdstanr
- Installs brms R package from CRAN
- Version control support

**[`fit()`](brmspy/brmspy.py:569-789)**
```python
def fit(
    formula: str,
    data: Union[dict, pd.DataFrame],
    priors: list = [],
    family: str = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    return_type: str = "idata",  # KEY PARAMETER
    **brm_args
) -> Union[InferenceData, brmsfit, BrmsFitResult]
```

**Key Parameters:**
- `return_type`: Controls output format
- `backend`: Stan backend ("cmdstanr", "rstan", "mock")
- `**brm_args`: Passed to brms::brm() (iter, warmup, chains, cores, etc.)

**[`_brmsfit_to_idata()`](brmspy/brmspy.py:503-567)**
- Uses posterior R package for conversion
- Extracts draws as pandas DataFrame
- Reshapes to (chain, draw) format
- Creates arviz.InferenceData via `az.from_dict()`

### Helper Functions

**[`get_brms_data()`](brmspy/brmspy.py:223-251)** - Load brms datasets

**[`get_brms_version()`](brmspy/brmspy.py:151-189)** - Check brms version

**[`_convert_python_to_R()`](brmspy/brmspy.py:254-282)** - Python → R conversion

**[`_convert_R_to_python()`](brmspy/brmspy.py:326-358)** - R → Python conversion

**[`_coerce_types()`](brmspy/brmspy.py:361-433)** - Type coercion for Stan

**[`get_stan_code()`](brmspy/brmspy.py:285-323)** - Generate Stan code (legacy)

## Dependencies

### Python Dependencies
```
Required:
- rpy2 ≥ 3.5.0    # Python-R bridge
- pandas ≥ 1.3.0  # DataFrames
- numpy ≥ 1.20.0  # Arrays

Optional:
- arviz ≥ 0.11.0  # InferenceData
```

### R Dependencies
```
Required:
- brms ≥ 2.20.0    # Bayesian regression
- cmdstanr         # Stan interface
- posterior        # Draw conversion

Installed by brms:
- rstan           # Legacy Stan interface (still used internally)
- Rcpp            # C++ integration
- loo, bayesplot  # Diagnostics
```

### System Dependencies
```
- R ≥ 4.0
- C++ compiler (for Stan)
- CmdStan (auto-installed via cmdstanr)
```

## Compatibility

### Python Versions
- **Supported:** 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- **Tested:** 3.12 (primary), others via CI
- **Minimum:** 3.8 (f-strings, type hints)

### Operating Systems
- **macOS:** ✅ Full support
- **Linux:** ✅ Full support
- **Windows:** ✅ Should work (not extensively tested)

### R Versions
- **Minimum:** R 4.0
- **Recommended:** R 4.3+
- **Note:** rpy2 compatibility may vary by R version

## Known Issues & Limitations

### 1. R Installation
- Requires R to be installed system-wide
- rpy2 must find R correctly
- May need R_HOME environment variable

### 2. rpy2 Compatibility
- Some platforms have rpy2 installation issues
- May need to install R development headers
- macOS: `brew install r`
- Ubuntu: `apt-get install r-base-dev`

### 3. CmdStan Compilation
- First-time Stan compilation can be slow (5-10 min)
- Requires C++ compiler
- Large disk space (CmdStan ~500MB)

### 4. Memory Usage
- R and Python both hold data in memory
- Large datasets may cause memory issues
- Consider chunking or sampling

### 5. Type Coercion Edge Cases
- Complex Stan types may not coerce correctly
- Custom Stan functions not tested
- Report issues if found

## Testing Strategy

### Test Structure
```
tests/
├── conftest.py           # Fixtures, brms detection
├── test_basic.py         # Unit tests (19 tests)
└── test_integration.py   # Integration tests (13 tests)
```

### Test Categories

**Unit Tests (no R required):**
- Module imports
- Data structure conversions
- Type coercion logic
- Error handling

**Integration Tests (require brms):**
- brms installation/version
- Data loading
- Model fitting (slow, marked with @pytest.mark.slow)
- Return type validation

### Running Tests
```bash
# Quick tests (no model fitting)
pytest tests/ -v -m "not slow"

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=brmspy --cov-report=html
```

### Coverage
- Current: 54% (116/248 lines)
- Core functions well tested
- Installation/error paths less tested (hard to mock R)


## Performance Considerations

### Compilation
- Stan models compiled on first use
- Cached by CmdStan (reused if unchanged)
- ~5-30 seconds typical compilation time

### Sampling
- Performance similar to native brms
- Parallel chains recommended: `chains=4, cores=4`
- cmdstanr backend slightly faster than rstan

### Memory
- Python + R both hold data
- arviz InferenceData more memory-efficient than R objects

### Type Conversion
- rpy2 conversions add ~50-100ms overhead
- Negligible compared to sampling time
- pandas2ri/numpy2ri well-optimized

## Future Enhancements

### Planned
- [ ] More comprehensive type tests
- [ ] Windows CI testing
- [ ] Performance benchmarks
- [ ] More arviz integration examples

### Considered
- Direct cmdstanr interface (bypass rpy2)
  - Pro: Faster, no R dependency
  - Con: Lose brms formula syntax
- Custom Stan code support
  - Pro: More flexibility
  - Con: Loses brms benefits
- Async sampling support
  - Pro: Non-blocking
  - Con: Complex implementation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## References

- [brms Documentation](https://paul-buerkner.github.io/brms/)
- [cmdstanr Documentation](https://mc-stan.org/cmdstanr/)
- [arviz Documentation](https://arviz-devs.github.io/arviz/)
- [rpy2 Documentation](https://rpy2.github.io/)
- [Stan Documentation](https://mc-stan.org/docs/)