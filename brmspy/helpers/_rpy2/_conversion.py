import re
from collections.abc import Callable
from typing import Any, Literal, cast

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from brmspy.helpers._rpy2._converters import (
    py_to_r,
    r_to_py,
)
from brmspy.helpers.log import log_warning
from brmspy.types.brms_results import IDBrm
from rpy2.rinterface_lib.sexp import Sexp

__all__ = ["py_to_r", "r_to_py"]


def _coerce_stan_types(stan_code: str, stan_data: dict) -> dict:
    """
    Coerce Python numeric types to match Stan data block requirements.

    Parses the Stan program's data block to determine variable types (int vs real)
    and automatically coerces Python data to match. Handles both old Stan syntax
    (`int Y[N]`) and new array syntax (`array[N] int Y`). Converts single-element
    arrays to scalars when appropriate.

    Parameters
    ----------
    stan_code : str
        Complete Stan program code containing a data block
    stan_data : dict
        Dictionary of data to pass to Stan, with keys matching Stan variable names

    Returns
    -------
    dict
        Type-coerced data dictionary with:
        - Integer types coerced to int/int64 where Stan expects int
        - Single-element arrays converted to scalars
        - Multi-element arrays preserved with correct dtype

    Notes
    -----
    **Stan Type Coercion:**

    Stan requires strict type matching:
    - `int` variables must receive integer values
    - `real` variables can receive floats
    - Arrays must have consistent element types

    **Syntax Support:**

    Old Stan syntax (pre-2.26):
    ```stan
    data {
      int N;
      int Y[N];
      real X[N];
    }
    ```

    New Stan syntax (2.26+):
    ```stan
    data {
      int N;
      array[N] int Y;
      array[N] real X;
    }
    ```

    **Scalar Coercion:**

    Single-element numpy arrays are automatically converted to scalars:
    - `np.array([5])` → `5`
    - `np.array([5.0])` → `5.0`

    Examples
    --------

    ```python
    stan_code = '''
    data {
        int N;
        array[N] int y;
        array[N] real x;
    }
    model {
        y ~ poisson_log(x);
    }
    '''

    # Python data with incorrect types
    data = {
        'N': 3.0,  # Should be int
        'y': np.array([1.5, 2.5, 3.5]),  # Should be int
        'x': np.array([0.1, 0.2, 0.3])  # OK as real
    }

    # Coerce to match Stan requirements
    coerced = _coerce_stan_types(stan_code, data)
    # Result: {'N': 3, 'y': array([1, 2, 3]), 'x': array([0.1, 0.2, 0.3])}
    ```

    See Also
    --------
    brmspy.brms.make_stancode : Generate Stan code from brms formula
    brmspy.brms.fit : Automatically applies type coercion during fitting
    """
    pat_data = re.compile(r"(?<=data {)[^}]*")
    pat_identifiers = re.compile(r"([\w]+)")

    # Extract the data block and separate lines
    data_lines = pat_data.findall(stan_code)[0].split("\n")

    # Remove comments, <>-style bounds and []-style data size declarations
    data_lines_no_comments = [l.split("//")[0] for l in data_lines]
    data_lines_no_bounds = [re.sub("<[^>]+>", "", l) for l in data_lines_no_comments]
    data_lines_no_sizes = [re.sub(r"\[[^>]+\]", "", l) for l in data_lines_no_bounds]

    # Extract identifiers and handle both old and new Stan syntax
    # Old: int Y; or int Y[N]; -> type is first identifier
    # New: array[N] int Y; -> type is second identifier (after 'array')
    identifiers = [pat_identifiers.findall(l) for l in data_lines_no_sizes]

    var_types = []
    var_names = []
    for tokens in identifiers:
        if len(tokens) == 0:
            continue
        # New syntax: array[...] type name
        if tokens[0] == "array" and len(tokens) >= 3:
            var_types.append(tokens[1])  # Type is second token
            var_names.append(tokens[-1])  # Name is last token
        # Old syntax: type name
        elif len(tokens) >= 2:
            var_types.append(tokens[0])  # Type is first token
            var_names.append(tokens[-1])  # Name is last token

    var_dict = dict(zip(var_names, var_types))

    # Coerce integers to int and 1-size arrays to scalars
    for k, v in stan_data.items():
        # Convert to numpy array if not already
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
            stan_data[k] = v

        # First, convert 1-size arrays to scalars
        if hasattr(v, "size") and v.size == 1 and hasattr(v, "ndim") and v.ndim > 0:
            v = v.item()
            stan_data[k] = v

        # Then coerce to int if Stan expects int
        if k in var_names and var_dict[k] == "int":
            # Handle both scalars and arrays
            if isinstance(v, (int, float, np.number)):  # Scalar
                stan_data[k] = int(v)
            elif isinstance(v, np.ndarray):  # Array
                stan_data[k] = v.astype(np.int64)

    return stan_data


def _brmsfit_get_posterior(
    brmsfit_obj: Sexp, **kwargs
) -> tuple[dict[str, np.ndarray], Sexp]:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # -------------------------------------------------
    # POSTERIOR (parameters) via posterior::as_draws_df
    # -------------------------------------------------
    as_draws_df = cast(Callable, ro.r("posterior::as_draws_df"))
    draws_r = as_draws_df(brmsfit_obj, **kwargs)

    with localconverter(ro.default_converter + pandas2ri.converter):
        df = pandas2ri.rpy2py(draws_r)

    chain_col = ".chain" if ".chain" in df.columns else "chain"
    draw_col = ".draw" if ".draw" in df.columns else "draw"

    df["draw_idx"] = df.groupby(chain_col)[draw_col].transform(
        lambda x: np.arange(len(x), dtype=int)
    )

    chains = np.sort(df[chain_col].unique())
    n_chains = len(chains)
    n_draws = int(df["draw_idx"].max()) + 1

    posterior_dict: dict[str, np.ndarray] = {}

    for col in df.columns:
        if col in (chain_col, draw_col, ".iteration", "draw_idx"):
            continue

        mat = (
            df.pivot(index="draw_idx", columns=chain_col, values=col)
            .sort_index(axis=0)
            .reindex(columns=chains)
            .to_numpy()
            .T
        )
        posterior_dict[col] = mat

    return posterior_dict, draws_r


def _brmsfit_get_response_names(brmsfit_obj) -> list[str]:
    import rpy2.robjects as ro

    # ------------------------------
    # RESPONSE NAMES via brmsterms()
    # ------------------------------
    resp_names: list[str] = []
    try:
        # Method 1: Use brmsterms to get response variable names
        r_code = """
        function(fit) {
            bterms <- brms::brmsterms(fit$formula)
            if (inherits(bterms, "mvbrmsterms")) {
                # Multivariate: extract response names from each term
                names(bterms$terms)
            } else {
                # Univariate: get the single response
                resp <- bterms$respform
                if (!is.null(resp)) {
                    all.vars(resp)[1]
                } else {
                    # Fallback: parse from formula
                    all.vars(fit$formula$formula)[1]
                }
            }
        }
        """
        get_resp_names = cast(Callable, ro.r(r_code))
        resp_r = get_resp_names(brmsfit_obj)
        resp_names = list(resp_r)
    except Exception as e:
        log_warning(
            f"[brmsfit_to_idata] Could not get response names via brmsterms: {e}"
        )

        # Fallback: try to extract from model formula directly
        try:
            r_fallback = """
            function(fit) {
                # Try to get response names from the model's data
                y <- brms::get_y(fit)
                if (is.matrix(y) || is.data.frame(y)) {
                    colnames(y)
                } else if (!is.null(names(y))) {
                    unique(names(y))[1]
                } else {
                    "y"
                }
            }
            """
            get_resp_fallback = cast(Callable, ro.r(r_fallback))
            resp_r = get_resp_fallback(brmsfit_obj)
            if hasattr(resp_r, "__iter__") and not isinstance(resp_r, str):
                resp_names = [str(r) for r in resp_r if r is not None]
            else:
                resp_names = [str(resp_r)]
        except Exception as e2:
            log_warning(f"[brmsfit_to_idata] Fallback also failed: {e2}")

    return resp_names


TypeDims = dict[str, list[str]]
TypeCoords = dict[str, np.ndarray]


def _is_unique(values) -> bool:
    """Return True if all values are unique (no duplicates)."""
    # np.unique is fine here; values are small-ish 1D vectors
    vals = np.asarray(values)
    return np.unique(vals).size == vals.size


def _get_obs_id_from_r_data(r_data, n_obs: int):
    """
    Decide obs_id for in-sample data from brmsfit$data.

    Priority:
    1. `_obs_id_` column if present and unique.
    2. rownames if present and unique.
    3. fallback: np.arange(n_obs).
    """
    import rpy2.robjects as ro

    fun_colnames = cast(Callable, ro.r("colnames"))
    fun_rownames = cast(Callable, ro.r("rownames"))
    colnames = list(cast(ro.ListVector, fun_colnames(r_data)))

    # 1) explicit obs_id column
    if "_obs_id_" in colnames:
        obs_col = np.asarray(r_data.rx2("_obs_id_"))
        if _is_unique(obs_col):
            return obs_col
        else:
            log_warning(
                "Column '_obs_id_' in brmsfit$data is not unique; "
                "falling back to rownames or sequential indices."
            )
    elif "obs_id" in colnames:
        obs_col = np.asarray(r_data.rx2("obs_id"))
        if _is_unique(obs_col):
            return obs_col
        else:
            log_warning(
                "Column 'obs_id' in brmsfit$data is not unique; "
                "falling back to rownames or sequential indices."
            )

    # 2) unique rownames
    rownames = np.asarray(fun_rownames(r_data))
    if len(rownames) == n_obs and _is_unique(rownames):
        return rownames

    # 3) fallback: 0-based integer index
    log_warning(
        "Unable to find a unique obs_id in brmsfit$data "
        "(no unique '_obs_id_'/'obs_id' column or rownames). "
        "Using sequential indices 0..N-1."
    )
    return np.arange(n_obs, dtype=np.int64)


def _get_obs_id_from_newdata(newdata: pd.DataFrame, n_obs: int):
    """
    Decide obs_id for newdata (out-of-sample).

    Priority:
    1. `obs_id` column if present and unique.
    2. newdata.index (with warning if not unique).
    """
    if "_obs_id_" in newdata.columns:
        obs_col = newdata["_obs_id_"].to_numpy()
        if _is_unique(obs_col):
            return obs_col
        else:
            log_warning(
                "Column '_obs_id_' in newdata is not unique; "
                "falling back to DataFrame index."
            )
    elif "obs_id" in newdata.columns:
        obs_col = newdata["obs_id"].to_numpy()
        if _is_unique(obs_col):
            return obs_col
        else:
            log_warning(
                "Column 'obs_id' in newdata is not unique; "
                "falling back to DataFrame index."
            )

    index_vals = newdata.index.to_numpy()
    if not _is_unique(index_vals):
        log_warning(
            "newdata.index is not unique; using it as obs_id anyway. "
            "This may cause ambiguous mapping in ArviZ."
        )
    return index_vals


def _brmsfit_get_dims_and_coords(
    brmsfit_obj,
    newdata: None | pd.DataFrame = None,
    resp_names: None | list[str] = None,
) -> tuple[TypeDims, TypeCoords]:
    """
    Infer dims/coords for ArviZ from a brmsfit object and optional newdata.

    Rules for obs_id:
    - If newdata is None:
        1) If `obs_id` column exists in `fit$data` and is unique: use that.
        2) Else, if rownames of `fit$data` are unique: use those.
        3) Else: use a sequential integer range [0, N).
    - If newdata is not None:
        1) If `obs_id` column exists in newdata and is unique: use that.
        2) Else: use newdata.index (with a warning if not unique).
    """
    import rpy2.robjects as ro

    fun_nrow = cast(Callable, ro.r("nrow"))

    if resp_names is None:
        resp_names = _brmsfit_get_response_names(brmsfit_obj)

    if newdata is None:
        # in-sample: look at brmsfit$data
        r_data = brmsfit_obj.rx2("data")
        n_obs = int(fun_nrow(r_data)[0])

        obs_id = _get_obs_id_from_r_data(r_data, n_obs)

    else:
        # out-of-sample: look at newdata
        n_obs = int(len(newdata))
        obs_id = _get_obs_id_from_newdata(newdata, n_obs)

    obs_id_arr = np.asarray(obs_id)

    dims: TypeDims = {}
    coords: TypeCoords = {}

    coords["obs_id"] = obs_id_arr

    for resp in resp_names:
        dims[resp] = ["obs_id"]

    # you can add more dims/coords for responses here later
    # e.g. multi-response mapping, etc.

    return dims, coords


def _brmsfit_get_observed_data(
    brmsfit_obj, resp_names: None | list[str] = None
) -> dict[str, np.ndarray]:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    if resp_names is None:
        resp_names = _brmsfit_get_response_names(brmsfit_obj)

    observed_data_dict: dict[str, np.ndarray] = {}
    n_obs = 0

    try:
        r_get_y = cast(Callable, ro.r("brms::get_y"))
        y_r = r_get_y(brmsfit_obj)

        with localconverter(ro.default_converter + pandas2ri.converter):
            y_py = pandas2ri.rpy2py(y_r)

        if isinstance(y_py, pd.DataFrame):
            n_obs = y_py.shape[0]
            if not resp_names:
                resp_names = list(y_py.columns)
            for resp in resp_names:
                if resp in y_py.columns:
                    observed_data_dict[resp] = y_py[resp].to_numpy()

        elif isinstance(y_py, pd.Series):
            n_obs = y_py.shape[0]
            if not resp_names:
                resp_names = [str(y_py.name) or "y"]
            observed_data_dict[resp_names[0]] = y_py.to_numpy()

        else:
            arr = np.asarray(y_py)
            if arr.ndim == 1:
                n_obs = arr.shape[0]
                if not resp_names:
                    resp_names = ["y"]
                observed_data_dict[resp_names[0]] = arr
            elif arr.ndim == 2:
                n_obs, k = arr.shape
                if not resp_names:
                    resp_names = [f"y_{j}" for j in range(k)]
                for j, resp in enumerate(resp_names):
                    observed_data_dict[resp] = arr[:, j]

    except Exception as e:
        log_warning(f"[brmsfit_to_idata] Could not extract observed data: {e}")

    return observed_data_dict


def _reshape_to_arviz(values: np.ndarray, n_chains: int, n_draws: int) -> np.ndarray:
    values = np.asarray(values)
    total = values.shape[0]
    expected = n_chains * n_draws
    if total != expected:
        raise ValueError(f"Expected {expected} rows (chains*draws), got {total}")
    new_shape = (n_chains, n_draws) + values.shape[1:]
    return values.reshape(new_shape)


def _brmsfit_get_counts(brmsfit_obj: Sexp) -> tuple[int, int]:
    """
    returns (ndraws, nchains)
    ndraws - draws per chain
    """
    import rpy2.robjects as ro

    fun_ndraws = cast(Callable, ro.r("posterior::ndraws"))
    fun_nchains = cast(Callable, ro.r("posterior::nchains"))

    ndraws = int(fun_ndraws(brmsfit_obj)[0])
    nchains = int(fun_nchains(brmsfit_obj)[0])

    ndraws = ndraws // nchains

    return nchains, ndraws


def _brmsfit_get_predict_generic(
    brmsfit_obj,
    function: Literal[
        "brms::posterior_predict",
        "brms::log_lik",
        "brms::posterior_linpred",
        "brms::posterior_epred",
    ] = "brms::posterior_predict",
    resp_names: None | list[str] = None,
    **kwargs,
) -> tuple[dict[str, np.ndarray], Sexp | dict[str, Sexp]]:
    import rpy2.robjects as ro

    if resp_names is None:
        resp_names = _brmsfit_get_response_names(brmsfit_obj)

    nchains, ndraws = _brmsfit_get_counts(brmsfit_obj)

    post_pred_dict: dict[str, np.ndarray] = {}

    r: dict[str, Sexp] | Sexp = ro.NULL
    try:
        r_pp_wrapper = cast(Callable, ro.r(function))

        if not resp_names:
            # No response names found - univariate default
            pp_r = r_pp_wrapper(brmsfit_obj, **kwargs)
            r = pp_r
            post_pred_dict["y"] = _reshape_to_arviz(np.asarray(pp_r), nchains, ndraws)

        elif len(resp_names) == 1:
            # Single response
            resp = resp_names[0]
            pp_r = r_pp_wrapper(
                brmsfit_obj, **kwargs, resp=resp
            )  # Pass as plain string
            r = pp_r
            post_pred_dict[resp] = _reshape_to_arviz(np.asarray(pp_r), nchains, ndraws)

        else:
            # Multivariate: loop over response names
            r = {}
            for resp in resp_names:
                pp_r = r_pp_wrapper(
                    brmsfit_obj, **kwargs, resp=resp
                )  # Pass as plain string!
                post_pred_dict[resp] = _reshape_to_arviz(
                    np.asarray(pp_r), nchains, ndraws
                )
                r[resp] = pp_r

    except Exception as e:
        log_warning(
            f"[brmsfit_to_idata] Could not extract posterior predictive/log_lik: {e}"
        )
        import traceback

        traceback.print_exc()

    return post_pred_dict, r


def _brmsfit_get_constant_data(
    brmsfit_obj,
    newdata: None | pd.DataFrame = None,
    resp_names: None | list[str] = None,
) -> dict[str, np.ndarray]:
    """
    Extract constant_data for ArviZ.

    - If newdata is None: use brmsfit$data.
    - Else: use the provided newdata.
    - Drop response columns and 'obs_id' (responses go to observed_data,
      obs_id is handled as a coord).
    - Return a dict[var_name -> np.ndarray] with length N (N = number of rows).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    if resp_names is None:
        resp_names = _brmsfit_get_response_names(brmsfit_obj)

    if newdata is None:
        # in-sample: use brmsfit$data (R data.frame) -> pandas.DataFrame
        r_data = brmsfit_obj.rx2("data")
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = pandas2ri.rpy2py(r_data)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    else:
        # out-of-sample: use newdata as given
        df = newdata.copy()

    # Ensure we don't accidentally mutate caller's frame
    df = df.copy()

    # Drop response variables if present
    drop_cols: set[str] = set(resp_names or [])

    # Drop obs_id column if present; obs_id is handled as a coord
    if "_obs_id_" in df.columns:
        df = df.set_index("_obs_id_", drop=True)

    keep_cols = [c for c in df.columns if c not in drop_cols]

    constant_data: dict[str, np.ndarray] = {}
    for col in keep_cols:
        # Just pass through whatever dtype it has; xarray can handle object too
        constant_data[col] = df[col].to_numpy()

    return constant_data


def _arviz_add_constant_data(
    idata: az.InferenceData,
    constant_data_dict: dict[str, np.ndarray],
    group_name: Literal["constant_data", "predictions_constant_data"] = "constant_data",
    obs_id: None | list[str] | np.ndarray = None,
) -> az.InferenceData:
    """
    Add a non-draw group (constant_data or predictions_constant_data) to an idata.

    Extracts obs_id coords directly from the existing idata. This avoids ArviZ's
    auto (chain, draw) dims and keeps the group purely 1D along obs_id.
    """
    if not constant_data_dict:
        return idata

    # ---- 1) Extract obs_id coords from any existing group ----
    if obs_id is None:
        for group in idata.groups():
            ds = idata[group]
            if ds is not None and "obs_id" in ds.coords:
                obs_id = ds.coords["obs_id"].values
                break

        if obs_id is None:
            raise ValueError(
                "Could not locate 'obs_id' in any existing idata group; "
                "cannot attach constant_data."
            )

    # ---- 2) Build dims & coords for the new constant group ----
    const_dims = {name: ["obs_id"] for name in constant_data_dict.keys()}
    const_coords = {"obs_id": obs_id}

    # ---- 3) Build a small InferenceData and extend ----
    if group_name == "constant_data":
        const_idata = az.from_dict(
            constant_data=constant_data_dict, coords=const_coords, dims=const_dims
        )
    else:
        const_idata = az.from_dict(
            predictions_constant_data=constant_data_dict,
            coords=const_coords,
            dims=const_dims,
        )

    idata.extend(const_idata)
    return idata


def _idata_add_resp_names_suffix(
    idata: az.InferenceData,
    suffix: str,
    resp_names: list[str],
) -> None:
    """
    In-place: append `suffix` to all variables in `resp_names` across all
    applicable InferenceData groups.

    Mutates `idata` directly.
    """
    if not suffix or not resp_names:
        return

    for group in idata.groups():
        ds = getattr(idata, group, None)
        if ds is None:
            continue

        rename_map = {
            resp: f"{resp}{suffix}" for resp in resp_names if resp in ds.data_vars
        }

        if rename_map:
            ds = ds.rename(rename_map)
            setattr(idata, group, ds)


def brmsfit_to_idata(brmsfit_obj, model_data=None) -> IDBrm:
    posterior_dict, _ = _brmsfit_get_posterior(brmsfit_obj)
    resp_names = _brmsfit_get_response_names(brmsfit_obj)
    dims, coords = _brmsfit_get_dims_and_coords(brmsfit_obj, resp_names=resp_names)
    observed_data_dict = _brmsfit_get_observed_data(brmsfit_obj, resp_names)
    post_pred_dict, _ = _brmsfit_get_predict_generic(
        brmsfit_obj, function="brms::posterior_predict", resp_names=resp_names
    )
    log_lik_dict, _ = _brmsfit_get_predict_generic(
        brmsfit_obj, function="brms::log_lik", resp_names=resp_names
    )
    constant_data_dict = _brmsfit_get_constant_data(
        brmsfit_obj, newdata=None, resp_names=resp_names
    )
    for name in constant_data_dict:
        if name not in dims:
            dims[name] = ["obs_id"]

    idata = az.from_dict(
        posterior=posterior_dict,
        posterior_predictive=post_pred_dict or None,
        log_likelihood=log_lik_dict or None,
        observed_data=observed_data_dict or None,
        coords=coords or None,
        constant_data=constant_data_dict or None,
        dims=dims or None,
    )

    return cast(IDBrm, idata)


def kwargs_r(kwargs: dict | None) -> dict:
    """
    Convert Python keyword arguments to R-compatible format.

    Convenience function that applies py_to_r() to all values in a
    keyword arguments dictionary, preparing them for R function calls.

    Parameters
    ----------
    kwargs : dict or None
        Dictionary of keyword arguments where values may be Python objects
        (dicts, lists, DataFrames, arrays, etc.)

    Returns
    -------
    dict
        Dictionary with same keys but R-compatible values, or empty dict if None

    Notes
    -----
    This is a thin wrapper around `py_to_r()` that operates on dictionaries.
    It's commonly used to prepare keyword arguments for R function calls via rpy2.

    Examples
    --------

    ```python
    from brmspy.helpers.conversion import kwargs_r
    import pandas as pd
    import numpy as np

    # Prepare kwargs for R function
    py_kwargs = {
        'data': pd.DataFrame({'y': [1, 2], 'x': [1, 2]}),
        'prior': {'b': [0, 1]},
        'chains': 4,
        'iter': 2000
    }

    r_kwargs = kwargs_r(py_kwargs)
    # All values converted to R objects
    # Can now call: r_function(**r_kwargs)
    ```

    See Also
    --------
    py_to_r : Underlying conversion function for individual values
    brmspy.brms.fit : Uses this to prepare user kwargs for R
    """
    if kwargs is None:
        return {}
    return {k: py_to_r(v) for k, v in kwargs.items()}
