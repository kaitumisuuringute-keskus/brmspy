import typing
import pandas as pd
import numpy as np
import re
import warnings
import xarray as xr
import arviz as az

import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri, numpy2ri, ListVector, DataFrame, StrVector
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

_brms = None

def _get_brms():
    """
    Lazy import of brms with helpful error message if not installed.
    
    Returns
    -------
    brms module
        The imported brms R package
    
    Raises
    ------
    ImportError
        If brms is not installed, with instructions for installation
    """
    global _brms
    if _brms is None:
        try:
            _brms = rpackages.importr("brms")
        except Exception as e:
            raise ImportError(
                "brms R package not found. Install it using:\n\n"
                "  import brmspy\n"
                "  brmspy.install_brms()  # for latest version\n\n"
                "Or install a specific version:\n"
                "  brmspy.install_brms(version='2.23.0')\n\n"
                "Or install manually in R:\n"
                "  install.packages('brms')\n"
            ) from e
    return _brms





def _convert_R_to_python(
    formula: str, 
    data: typing.Union[dict, pd.DataFrame], 
    family: str
) -> dict:
    """
    Convert R data structures from brms to Python dictionaries.
    
    Calls brms::make_standata() and converts the result to Python.
    
    Parameters
    ----------
    formula : str
        brms formula specification
    data : dict or pd.DataFrame
        Model data
    family : str
        Distribution family
    
    Returns
    -------
    dict
        Stan data as Python dictionary
    """
    brms = _get_brms()
    # Call brms to preprocess the data; returns an R ListVector
    model_data = brms.make_standata(formula, data, family=family)
    
    # Convert R objects to Python/pandas/numpy
    # We use a context manager because it conflicts with prior creation
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        model_data = dict(model_data.items())
    return model_data


def _coerce_types(stan_code: str, stan_data: dict) -> dict:
    """
    Coerce Python types to match Stan's type requirements.
    
    Stan has strict type requirements (int vs float). This function parses
    the Stan data block to determine required types and coerces the data
    accordingly.
    
    Parameters
    ----------
    stan_code : str
        Generated Stan code
    stan_data : dict
        Data dictionary to coerce
    
    Returns
    -------
    dict
        Type-coerced data dictionary
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

def _convert_python_to_R(data: typing.Union[dict, pd.DataFrame]):
    """
    Convert Python data structures to R objects that brms can handle.
    
    Parameters
    ----------
    data : dict or pd.DataFrame
        Python data to convert
    
    Returns
    -------
    R object
        R list (from dict) or R data.frame (from DataFrame)
    
    Raises
    ------
    ValueError
        If data type is not supported
    """
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        if isinstance(data, pd.DataFrame):
            return DataFrame(data)
        elif isinstance(data, dict):
            return ListVector(data)
        else:
            raise ValueError(
                f"Data should be either a pandas DataFrame or a dictionary, "
                f"got {type(data).__name__}"
            )




def brmsfit_to_idata(brmsfit_obj, model_data=None):
    """
    Convert R object to a complete arviz InferenceData object.
    Includes Posterior, Posterior Predictive, Log Likelihood, and Observed Data.
    """
    # 1. SETUP: Essential R packages
    try:
        # We only import these to ensure they are installed/loaded in R
        importr('posterior')
        importr('brms')
        base = importr('base')
    except Exception as e:
        raise ImportError(f"Required R packages (brms, posterior) not found. Error: {e}")

    # =========================================================================
    # GROUP 1: POSTERIOR (Parameters)
    # =========================================================================
    # Safely get the as_draws_df function
    as_draws_df = ro.r('posterior::as_draws_df')
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
        r_posterior_predict = ro.r('brms::posterior_predict')
        r_log_lik = ro.r('brms::log_lik')
        
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
        # Extract data from the fit object: fit$data
        r_data = base.getElement(brmsfit_obj, "data")
        
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
    
    return idata






def _reshape_r_prediction_to_arviz(r_matrix, brmsfit_obj, obs_coords=None):
    """
    Internal helper to reshape R prediction matrices (Draws x Obs) 
    into ArviZ 3D arrays (Chains x Draws x Obs).
    """
    # 1. Get dimensions from the model
    # We use R functions to be safe about how brms stored the fit
    try:
        r_nchains = ro.r('brms::nchains')
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


def brms_epred_to_idata(r_epred_obj, brmsfit_obj, newdata=None, var_name="epred"):
    """
    Converts result of brms::posterior_epred to arviz.InferenceData.
    
    The result is stored in the 'posterior' group (as it represents the 
    expected value/mean), or you could put it in a custom group.
    """
    # Determine coordinates from newdata if available
    obs_coords = None
    if newdata is not None and isinstance(newdata, pd.DataFrame):
        # Use DataFrame index if it's meaningful, otherwise default range
        obs_coords = newdata.index.values

    data_3d, coords, dims = _reshape_r_prediction_to_arviz(
        r_epred_obj, brmsfit_obj, obs_coords
    )

    # Create DataArray
    da = xr.DataArray(data_3d, coords=coords, dims=dims, name=var_name)

    # Store in 'posterior' group as it is the Expected Value (mu)
    # Alternatively, often stored in 'predictions' or 'posterior_predictive' 
    # depending on your specific preference. 
    # Here we use 'posterior' to distinguish it from noisy 'posterior_predictive'.
    return az.InferenceData(
        posterior=da.to_dataset()
    )


def brms_predict_to_idata(r_predict_obj, brmsfit_obj, newdata=None, var_name="y"):
    """
    Converts result of brms::posterior_predict to arviz.InferenceData.
    
    The result is stored in the 'posterior_predictive' group.
    """
    # Determine coordinates
    obs_coords = None
    if newdata is not None and isinstance(newdata, pd.DataFrame):
        obs_coords = newdata.index.values

    data_3d, coords, dims = _reshape_r_prediction_to_arviz(
        r_predict_obj, brmsfit_obj, obs_coords
    )

    # Create DataArray
    da = xr.DataArray(data_3d, coords=coords, dims=dims, name=var_name)

    # Store in standard ArviZ location for predictions
    return az.InferenceData(
        posterior_predictive=da.to_dataset()
    )