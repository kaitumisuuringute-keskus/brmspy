import typing
import pandas as pd
import numpy as np
import re
import xarray as xr
import arviz as az

import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri, numpy2ri, ListVector, DataFrame, StrVector
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import vectors

from rpy2.robjects.functions import SignatureTranslatedFunction

from brmspy.helpers import singleton
from brmspy.types import IDFit, PriorSpec



def _coerce_stan_types(stan_code: str, stan_data: dict) -> dict:
    """
    Coerce Python types to match Stan type requirements.
    
    Parses Stan data block to determine required types (int vs float)
    and coerces data accordingly. Handles both old and new Stan array syntax.
    
    Parameters
    ----------
    stan_code : str
        Stan program code
    stan_data : dict
        Data dictionary
    
    Returns
    -------
    dict
        Type-coerced data
    """
    pat_data = re.compile(r'(?<=data {)[^}]*')
    pat_identifiers = re.compile(r'([\w]+)')

    # Extract the data block and separate lines
    data_lines = pat_data.findall(stan_code)[0].split('\n')
    
    # Remove comments, <>-style bounds and []-style data size declarations
    data_lines_no_comments = [l.split('//')[0] for l in data_lines]
    data_lines_no_bounds = [re.sub('<[^>]+>', '', l) for l in data_lines_no_comments]
    data_lines_no_sizes = [re.sub(r'\[[^>]+\]', '', l) for l in data_lines_no_bounds]

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
        if tokens[0] == 'array' and len(tokens) >= 3:
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
        if hasattr(v, 'size') and v.size == 1 and hasattr(v, 'ndim') and v.ndim > 0:
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


def brmsfit_to_idata(brmsfit_obj, model_data=None) -> IDFit:
    """
    Convert brmsfit R object to arviz InferenceData.
    
    Includes posterior, posterior_predictive, log_likelihood, and observed_data groups.
    Handles proper chain/draw indexing for arviz compatibility.
    
    Parameters
    ----------
    brmsfit_obj : R brmsfit object
        Fitted model from brms::brm()
    model_data : optional
        Additional model data (currently unused)
    
    Returns
    -------
    arviz.InferenceData
        Complete InferenceData with all groups
    """

    # =========================================================================
    # GROUP 1: POSTERIOR (Parameters)
    # =========================================================================
    # Safely get the as_draws_df function
    as_draws_df = typing.cast(typing.Callable, ro.r('posterior::as_draws_df'))
    draws_r = as_draws_df(brmsfit_obj)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = pandas2ri.rpy2py(draws_r)
            
    # Handle Chain/Draw Indexing
    chain_col = '.chain' if '.chain' in df.columns else 'chain'
    draw_col = '.draw' if '.draw' in df.columns else 'draw'
    
    # Create a clean 0..N index for draws within each chain
    df['draw_idx'] = df.groupby(chain_col)[draw_col].transform(lambda x: np.arange(len(x)))
    
    chains = df[chain_col].unique()
    n_chains = len(chains)
    n_draws = df['draw_idx'].max() + 1
    
    # Helper to reshape flat arrays (total_draws, ...) -> (chains, draws, ...)
    def reshape_to_arviz(values):
        # values shape: (total_draws, n_obs)
        # Reshape to (n_chains, n_draws, n_obs)
        new_shape = (n_chains, n_draws) + values.shape[1:]
        return values.reshape(new_shape)

    posterior_dict = {}
    for col in df.columns:
        if col not in [chain_col, draw_col, '.iteration', 'draw_idx']:
            # Pivot ensures we respect the chain/draw structure explicitly
            mat = df.pivot(index='draw_idx', columns=chain_col, values=col)
            posterior_dict[col] = mat.values.T 

    # =========================================================================
    # GROUP 2 & 3: POSTERIOR PREDICTIVE & LOG LIKELIHOOD
    # =========================================================================
    post_pred_dict = {}
    log_lik_dict = {}

    try:
        # Get functions explicitly to avoid AttributeError
        r_posterior_predict = typing.cast(typing.Callable, ro.r('brms::posterior_predict'))
        r_log_lik = typing.cast(typing.Callable, ro.r('brms::log_lik'))
        
        # 1. Posterior Predictive
        # Returns matrix: (Total_Draws x N_Obs)
        pp_r = r_posterior_predict(brmsfit_obj)
        pp_mat = np.array(pp_r)
        post_pred_dict['y'] = reshape_to_arviz(pp_mat)
        
        # 2. Log Likelihood
        ll_r = r_log_lik(brmsfit_obj)
        ll_mat = np.array(ll_r)
        log_lik_dict['y'] = reshape_to_arviz(ll_mat)
        
    except Exception as e:
        print(f"Warning: Could not extract posterior predictive/log_lik. {e}")
        pass

    # =========================================================================
    # GROUP 4: OBSERVED DATA
    # =========================================================================
    observed_data_dict = {}
    coords = None
    dims = None
    
    try:
        _base = singleton._get_base()
        # Extract data from the fit object: fit$data
        if _base:
            r_data = _base.getElement(brmsfit_obj, "data")
        else:
            raise Exception("Base uninitialized (Should not happen if _get_brms was done)!")
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_data = pandas2ri.rpy2py(r_data)
        
        # Heuristic: The response variable is usually the first column 
        # (brms rearranges internal data frame to put response first)
        if df_data is not None and not df_data.empty:
            resp_var = df_data.columns[0]
            observed_data_dict['y'] = df_data[resp_var].values
            
            # Setup coordinates for ArviZ
            coords = {'obs_id': np.arange(len(df_data))}
            dims = {'y': ['obs_id']}
        
    except Exception as e:
        print(f"Warning: Could not extract observed data. {e}")

    # =========================================================================
    # CREATE INFERENCE DATA
    # =========================================================================
    idata = az.from_dict(
        posterior=posterior_dict,
        posterior_predictive=post_pred_dict if post_pred_dict else None,
        log_likelihood=log_lik_dict if log_lik_dict else None,
        observed_data=observed_data_dict if observed_data_dict else None,
        coords=coords,
        dims=dims
    )
    
    return typing.cast(IDFit, idata)





def _reshape_r_prediction_to_arviz(r_matrix, brmsfit_obj, obs_coords=None):
    """
    Reshape R prediction matrix to arviz format.
    
    Converts (Total_Draws x Observations) to (Chains x Draws x Observations).
    
    Parameters
    ----------
    r_matrix : R matrix
        Prediction matrix from brms
    brmsfit_obj : R brmsfit object
        Fitted model
    obs_coords : array-like, optional
        Observation coordinates
    
    Returns
    -------
    tuple
        (reshaped_data, coords, dims)
    """
    # 1. Get dimensions from the model
    # We use R functions to be safe about how brms stored the fit
    try:
        r_nchains = typing.cast(typing.Callable, ro.r('brms::nchains'))
        n_chains = int(r_nchains(brmsfit_obj)[0])
    except Exception:
        # Fallback if brms::nchains fails
        n_chains = 4 

    # 2. Convert R matrix to Numpy
    # Shape is (Total_Draws, N_Observations)
    mat = np.array(r_matrix)
    total_draws, n_obs = mat.shape
    
    # 3. Calculate draws per chain
    n_draws = total_draws // n_chains

    # 4. Reshape
    # brms/rstan usually stacks chains: [Chain1_Draws, Chain2_Draws, ...]
    # So we reshape to (n_chains, n_draws, n_obs)
    reshaped_data = mat.reshape((n_chains, n_draws, n_obs))

    # 5. Create Coordinates
    if obs_coords is None:
        obs_coords = np.arange(n_obs)
        
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_draws),
        "obs_id": obs_coords
    }
    
    return reshaped_data, coords, ["chain", "draw", "obs_id"]

def generic_pred_to_idata(r_pred_obj, brmsfit_obj, newdata=None, var_name="pred", az_name="posterior"):
    """
    Convert generic brms prediction to arviz InferenceData.
    
    Generic converter for prediction functions.
    
    Parameters
    ----------
    r_pred_obj : R matrix
        Prediction matrix from brms
    brmsfit_obj : R brmsfit object
        Fitted model
    newdata : pd.DataFrame, optional
        Data for predictions
    var_name : str, default="pred"
        Variable name in InferenceData
    az_name : str, default="posterior"
        Group name in InferenceData
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with specified group
    """
    # Determine coordinates from newdata if available
    obs_coords = None
    if newdata is not None and isinstance(newdata, pd.DataFrame):
        # Use DataFrame index if it's meaningful, otherwise default range
        obs_coords = newdata.index.values

    data_3d, coords, dims = _reshape_r_prediction_to_arviz(
        r_pred_obj, brmsfit_obj, obs_coords
    )

    # Create DataArray
    da = xr.DataArray(data_3d, coords=coords, dims=dims, name=var_name)

    # Store in 'posterior' group as it is the Expected Value (mu)
    # Alternatively, often stored in 'predictions' or 'posterior_predictive' 
    # depending on your specific preference. 
    # Here we use 'posterior' to distinguish it from noisy 'posterior_predictive'.
    params = {
        az_name: da.to_dataset()
    }
    return az.InferenceData(**params, warn_on_custom_groups=False)

def brms_epred_to_idata(r_epred_obj, brmsfit_obj, newdata=None, var_name="epred"):
    """
    Convert brms::posterior_epred result to arviz InferenceData.
    
    Stores expected values in 'posterior' group.
    """
    return generic_pred_to_idata(r_epred_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="posterior")


def brms_predict_to_idata(r_predict_obj, brmsfit_obj, newdata=None, var_name="y"):
    """
    Convert brms::posterior_predict result to arviz InferenceData.
    
    Stores predictions in 'posterior_predictive' group.
    """
    return generic_pred_to_idata(r_predict_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="posterior_predictive")

def brms_linpred_to_idata(r_linpred_obj, brmsfit_obj, newdata=None, var_name="linpred"):
    """
    Convert brms::posterior_linpred result to arviz InferenceData.
    
    Stores linear predictor in 'predictions' group.
    """
    return generic_pred_to_idata(r_linpred_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="predictions")

def brms_log_lik_to_idata(r_log_lik_obj, brmsfit_obj, newdata=None, var_name="log_lik"):
    """
    Convert brms::log_lik result to arviz InferenceData.
    
    Stores log-likelihood in 'log_likelihood' group.
    """
    return generic_pred_to_idata(r_log_lik_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="log_likelihood")




from collections.abc import Mapping, Sequence

def py_to_r(obj):
    """
    Convert arbitrary Python objects into R objects.

    - dict -> R named list (ListVector), recursively
    - list/tuple:
        * list of dicts -> R list of named lists
        * otherwise -> let rpy2 default converter handle
    - numpy arrays, pandas objects, scalars, strings, etc:
        -> let rpy2 default converter handle
    """
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        if obj is None:
            return ro.NULL

        if isinstance(obj, pd.DataFrame):
            return cv.py2rpy(obj)

        if isinstance(obj, Mapping):
            converted = {str(k): py_to_r(v) for k, v in obj.items()}
            return ListVector(converted)

        # 2) list / tuple: inspect contents
        if isinstance(obj, (list, tuple)):
            if not obj:
                return ListVector({})

            if all(isinstance(el, Mapping) for el in obj):
                # R lists are usually named or indexed; use 1-based index names
                converted = {str(i + 1): py_to_r(el) for i, el in enumerate(obj)}
                return ListVector(converted)

            # mixed / other lists: let rpy2 decide (vectors, lists, etc.)
            return cv.py2rpy(obj)

        if isinstance(obj, np.ndarray):
            return cv.py2rpy(obj)

        # everything else: trust rpy2's default converter
        return cv.py2rpy(obj)

    
def r_to_py(obj):
    """
    Generic R → Python converter.

    Rules:
    - NULL → None
    - Named ListVector → dict
    - Unnamed ListVector → list
    - Atomic Vector length 1 → Python scalar
    - Atomic Vector length >1 → Python list of scalars
    - Other objects → fallback: Python repr string
      (safe for formulas, language objects, etc.)
    """
    if obj is ro.NULL:
        return None

    # 1) Atomic vectors -------------------------------------------------------
    if isinstance(obj, vectors.Vector) and not isinstance(obj, ListVector):
        # length 1 → scalar
        if len(obj) == 1:
            # Try default R→Python conversion
            with localconverter(default_converter) as cv:
                py = cv.rpy2py(obj[0])
            return py

        # length >1 → list of scalars
        out = []
        for el in obj:
            with localconverter(default_converter) as cv:
                py = cv.rpy2py(el)
            out.append(py)
        return out

    # 2) ListVectors (named/unnamed) -----------------------------------------
    if isinstance(obj, ListVector):
        names = list(obj.names) if obj.names is not ro.NULL else None

        # Named list → dict
        if names and any(n is not ro.NULL and n != "" for n in names):
            result = {}
            for name in names:
                key = str(name) if name not in (None, "") else None
                result[key] = r_to_py(obj.rx2(name))
            return result

        # Unnamed → list
        return [r_to_py(el) for el in obj]

    # 3) Language objects, formulas, calls, functions -------------------------
    # rpy2 wraps these in LangVector, Formula, or other types.
    if isinstance(obj, (ro.Formula, ro.language.LangVector, SignatureTranslatedFunction)):
        # Return a plain descriptive string (safe fallback)
        return str(obj)

    # 4) Anything else → let rpy2 convert or stringify ------------------------
    try:
        with localconverter(default_converter) as cv:
            return cv.rpy2py(obj)
    except Exception:
        return str(obj)
    
def kwargs_r(kwargs: typing.Optional[typing.Dict]) -> typing.Dict:
    if kwargs is None:
        return {}
    return {k: py_to_r(v) for k, v in kwargs.items()}


