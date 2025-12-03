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
from brmspy.helpers.log import log_warning
from brmspy.types import IDFit, PriorSpec, RListVectorExtension



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
    Convert brmsfit R object to ArviZ InferenceData format.
    
    Comprehensive conversion that extracts all MCMC diagnostics, predictions,
    and observed data from a brms fitted model into ArviZ's InferenceData format.
    Handles proper chain/draw indexing and coordinates for ArviZ compatibility.
    
    Parameters
    ----------
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted Bayesian model from R's brms::brm() function
    model_data : dict or DataFrame, optional
        Additional model data (currently unused, reserved for future use)
    
    Returns
    -------
    arviz.InferenceData
        Complete InferenceData object containing:
        - **posterior**: Model parameters (β, σ, etc.) with shape (chains, draws, ...)
        - **posterior_predictive**: Predicted outcomes y with observation noise
        - **log_likelihood**: Log-likelihood values for LOO-CV and model comparison
        - **observed_data**: Original response variable from training data
        - **coords**: Observation IDs and other coordinates
        - **dims**: Dimension labels for each variable
    
    Notes
    -----
    **InferenceData Groups:**
    
    1. **Posterior**: Parameter samples from MCMC
       - Includes all model parameters (intercepts, slopes, sigmas, etc.)
       - Shape: (n_chains, n_draws) for scalars, (n_chains, n_draws, ...) for arrays
       - Extracted via posterior::as_draws_df()
    
    2. **Posterior Predictive**: Predictions including observation noise
       - Generated via brms::posterior_predict()
       - Useful for posterior predictive checks
       - Shape: (n_chains, n_draws, n_obs)
    
    3. **Log Likelihood**: Pointwise log-likelihood values
       - Generated via brms::log_lik()
       - Required for LOO-CV and WAIC model comparison
       - Shape: (n_chains, n_draws, n_obs)
    
    4. **Observed Data**: Original response variable
       - Extracted from fit$data
       - Uses first column (brms convention for response variable)
       - Shape: (n_obs,)
    
    **Reshaping Strategy:**
    
    brms/rstan return flat arrays with shape (total_draws, ...) where
    total_draws = n_chains × n_draws. This function reshapes to ArviZ's
    expected format (n_chains, n_draws, ...) by:
    1. Tracking chain IDs from .chain column
    2. Computing per-chain draw indices
    3. Pivoting and reshaping to 3D arrays
    
    Examples
    --------

    ```python
    from brmspy import fit
    from brmspy.helpers.conversion import brmsfit_to_idata
    import arviz as az
    
    # Fit a model (returns InferenceData directly)
    result = fit("y ~ x", data=pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]}))
    idata = result.idata
    
    # Or manually convert brmsfit object
    # brmsfit_r = ...  # R brmsfit object
    # idata = brmsfit_to_idata(brmsfit_r)
    
    # Use ArviZ for diagnostics
    az.summary(idata, var_names=["b_Intercept", "b_x"])
    az.plot_trace(idata)
    az.plot_posterior(idata, var_names=["sigma"])
    ```

    ```python
    # Access different groups
    idata.posterior  # Parameter samples
    idata.posterior_predictive  # Predicted outcomes with noise
    idata.log_likelihood  # For model comparison
    idata.observed_data  # Original data
    
    # Model comparison with LOO-CV
    az.loo(idata)
    az.waic(idata)
    ```

    See Also
    --------
    brmspy.brms.fit : High-level fitting function (returns InferenceData)
    brms_epred_to_idata : Convert expected value predictions
    brms_predict_to_idata : Convert posterior predictions
    arviz.InferenceData : ArviZ's data structure documentation
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
        log_warning(f"Could not extract posterior predictive/log_lik. {e}")
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
        log_warning(f"Could not extract observed data. {e}")

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
    Reshape brms prediction matrix from R to ArviZ-compatible format.
    
    Converts flat prediction matrix (total_draws × n_obs) to 3D array
    (n_chains × n_draws × n_obs) with proper coordinates and dimension names
    for ArviZ InferenceData objects.
    
    Parameters
    ----------
    r_matrix : rpy2 R matrix
        Prediction matrix from brms functions like posterior_predict(),
        posterior_epred(), etc. Shape: (total_draws, n_observations)
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model used to extract chain information
    obs_coords : array-like, optional
        Custom observation coordinate values (e.g., time points, IDs)
        Default: np.arange(n_obs)
    
    Returns
    -------
    tuple of (ndarray, dict, list)
        - **reshaped_data**: 3D numpy array with shape (n_chains, n_draws, n_obs)
        - **coords**: Dictionary with coordinate arrays for 'chain', 'draw', 'obs_id'
        - **dims**: List of dimension names ['chain', 'draw', 'obs_id']
    
    Notes
    -----
    **Reshaping Logic:**
    
    brms/rstan stack MCMC chains sequentially in the output matrix:
    ```
    [Chain1_Draw1, Chain1_Draw2, ..., Chain2_Draw1, Chain2_Draw2, ...]
    ```
    
    This function reshapes to ArviZ's expected format:
    ```
    (n_chains, n_draws_per_chain, n_observations)
    ```
    
    **Chain Detection:**
    
    Number of chains is extracted from the brmsfit object using brms::nchains().
    If extraction fails, falls back to default of 4 chains (cmdstanr default).
    
    Examples
    --------

    ```python
    import numpy as np
    from brmspy.helpers.conversion import _reshape_r_prediction_to_arviz
    
    # Simulate R prediction matrix (1000 total draws, 50 observations)
    # Assuming 4 chains × 250 draws = 1000 total draws
    r_matrix = np.random.randn(1000, 50)
    
    # Reshape with default coordinates
    data_3d, coords, dims = _reshape_r_prediction_to_arviz(
        r_matrix, brmsfit_obj
    )
    print(data_3d.shape)  # (4, 250, 50)
    print(coords.keys())  # dict_keys(['chain', 'draw', 'obs_id'])
    ```

    ```python
    # Custom observation coordinates (e.g., time series)
    import pandas as pd
    
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data_3d, coords, dims = _reshape_r_prediction_to_arviz(
        r_matrix, brmsfit_obj, obs_coords=dates
    )
    print(coords['obs_id'])  # DatetimeIndex with dates
    ```

    See Also
    --------
    generic_pred_to_idata : Uses this for creating InferenceData
    brmsfit_to_idata : Main conversion function for fitted models
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
    Generic converter for brms prediction matrices to ArviZ InferenceData.
    
    Flexible conversion function that handles various brms prediction types
    (posterior_predict, posterior_epred, posterior_linpred, log_lik) and
    stores them in appropriate InferenceData groups with proper structure.
    
    Parameters
    ----------
    r_pred_obj : rpy2 R matrix
        Prediction matrix from any brms prediction function
        Shape: (total_draws, n_observations)
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model for extracting chain information
    newdata : pd.DataFrame, optional
        New data used for predictions. If provided, DataFrame index
        is used for observation coordinates
    var_name : str, default="pred"
        Name for the variable in the InferenceData dataset
    az_name : str, default="posterior"
        InferenceData group name. Common values:
        - "posterior": For expected values (epred)
        - "posterior_predictive": For predictions with noise (predict)
        - "predictions": For linear predictor (linpred)
        - "log_likelihood": For log-likelihood values
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with single group containing reshaped predictions
        as xarray DataArray with proper coordinates and dimensions
    
    Notes
    -----
    **InferenceData Group Selection:**
    
    Different prediction types should use appropriate groups:
    - Expected values (epred): 'posterior' - deterministic E[Y|X]
    - Predictions (predict): 'posterior_predictive' - with observation noise
    - Linear predictor (linpred): 'predictions' - before link function
    - Log-likelihood: 'log_likelihood' - for model comparison
    
    **Coordinates:**
    
    If newdata is a DataFrame, uses its index as observation coordinates.
    This preserves meaningful labels (dates, IDs, etc.) in ArviZ plots.
    
    Examples
    --------

    ```python
    import pandas as pd
    from brmspy.helpers.conversion import generic_pred_to_idata
    
    # Assume we have fitted model and prediction matrix
    # r_epred = brms::posterior_epred(brmsfit, newdata=test_df)
    
    test_df = pd.DataFrame({'x': [1, 2, 3]}, index=['A', 'B', 'C'])
    
    idata = generic_pred_to_idata(
        r_pred_obj=r_epred,
        brmsfit_obj=brmsfit,
        newdata=test_df,
        var_name="expected_y",
        az_name="posterior"
    )
    
    # Access predictions
    print(idata.posterior['expected_y'].dims)  # ('chain', 'draw', 'obs_id')
    print(idata.posterior['expected_y'].coords['obs_id'])  # ['A', 'B', 'C']
    ```

    See Also
    --------
    brms_epred_to_idata : Convenience wrapper for posterior_epred
    brms_predict_to_idata : Convenience wrapper for posterior_predict
    brms_linpred_to_idata : Convenience wrapper for posterior_linpred
    brms_log_lik_to_idata : Convenience wrapper for log_lik
    _reshape_r_prediction_to_arviz : Internal reshaping function
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
    Convert brms::posterior_epred result to ArviZ InferenceData.
    
    Convenience wrapper for converting expected value predictions (posterior_epred)
    to InferenceData format. Stores in 'posterior' group as deterministic
    expected values E[Y|X] without observation noise.
    
    Parameters
    ----------
    r_epred_obj : rpy2 R matrix
        Result from brms::posterior_epred()
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model
    newdata : pd.DataFrame, optional
        New data used for predictions
    var_name : str, default="epred"
        Variable name in InferenceData
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with 'posterior' group containing expected values
    
    Notes
    -----
    **posterior_epred** computes the expected value of the posterior predictive
    distribution (i.e., the mean outcome for given predictors):
    - For linear regression: E[Y|X] = μ = X·β
    - For Poisson regression: E[Y|X] = exp(X·β)
    - For logistic regression: E[Y|X] = logit⁻¹(X·β)
    
    This is stored in the 'posterior' group (not 'posterior_predictive')
    because it represents deterministic expected values, not noisy predictions.
    
    See Also
    --------
    brmspy.brms.posterior_epred : High-level wrapper that calls this
    generic_pred_to_idata : Generic conversion function
    brms_predict_to_idata : For predictions with observation noise
    """
    return generic_pred_to_idata(r_epred_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="posterior")


def brms_predict_to_idata(r_predict_obj, brmsfit_obj, newdata=None, var_name="y"):
    """
    Convert brms::posterior_predict result to ArviZ InferenceData.
    
    Convenience wrapper for converting posterior predictions (posterior_predict)
    to InferenceData format. Stores in 'posterior_predictive' group as
    predictions including observation-level noise.
    
    Parameters
    ----------
    r_predict_obj : rpy2 R matrix
        Result from brms::posterior_predict()
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model
    newdata : pd.DataFrame, optional
        New data used for predictions
    var_name : str, default="y"
        Variable name in InferenceData
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with 'posterior_predictive' group containing predictions
    
    Notes
    -----
    **posterior_predict** generates predictions from the posterior predictive
    distribution, including observation-level noise:
    - For linear regression: Y ~ Normal(μ, σ)
    - For Poisson regression: Y ~ Poisson(λ)
    - For logistic regression: Y ~ Bernoulli(p)
    
    These predictions include all sources of uncertainty (parameter and observation)
    and are useful for:
    - Posterior predictive checks
    - Generating realistic synthetic data
    - Assessing model fit to observed data
    
    See Also
    --------
    brmspy.brms.posterior_predict : High-level wrapper that calls this
    generic_pred_to_idata : Generic conversion function
    brms_epred_to_idata : For expected values without noise
    """
    return generic_pred_to_idata(r_predict_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="posterior_predictive")

def brms_linpred_to_idata(r_linpred_obj, brmsfit_obj, newdata=None, var_name="linpred"):
    """
    Convert brms::posterior_linpred result to ArviZ InferenceData.
    
    Convenience wrapper for converting linear predictor values (posterior_linpred)
    to InferenceData format. Stores in 'predictions' group as linear predictor
    values before applying the link function.
    
    Parameters
    ----------
    r_linpred_obj : rpy2 R matrix
        Result from brms::posterior_linpred()
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model
    newdata : pd.DataFrame, optional
        New data used for predictions
    var_name : str, default="linpred"
        Variable name in InferenceData
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with 'predictions' group containing linear predictor
    
    Notes
    -----
    **posterior_linpred** returns the linear predictor η = X·β before
    applying the link function:
    - For linear regression: linpred = μ (same as epred since link is identity)
    - For Poisson regression: linpred = log(λ), epred = λ
    - For logistic regression: linpred = logit(p), epred = p
    
    The linear predictor is useful for:
    - Understanding the scale of effects before transformation
    - Diagnosing model specification issues
    - Custom post-processing with different link functions
    
    See Also
    --------
    brmspy.brms.posterior_linpred : High-level wrapper that calls this
    generic_pred_to_idata : Generic conversion function
    brms_epred_to_idata : For expected values on response scale
    """
    return generic_pred_to_idata(r_linpred_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="predictions")

def brms_log_lik_to_idata(r_log_lik_obj, brmsfit_obj, newdata=None, var_name="log_lik"):
    """
    Convert brms::log_lik result to ArviZ InferenceData.
    
    Convenience wrapper for converting pointwise log-likelihood values (log_lik)
    to InferenceData format. Stores in 'log_likelihood' group for use in
    model comparison and diagnostics.
    
    Parameters
    ----------
    r_log_lik_obj : rpy2 R matrix
        Result from brms::log_lik()
    brmsfit_obj : rpy2 R object (brmsfit)
        Fitted model
    newdata : pd.DataFrame, optional
        New data for log-likelihood calculation
    var_name : str, default="log_lik"
        Variable name in InferenceData
    
    Returns
    -------
    arviz.InferenceData
        InferenceData with 'log_likelihood' group
    
    Notes
    -----
    **log_lik** computes pointwise log-likelihood values for each observation,
    which are essential for:
    
    - **LOO-CV**: Leave-one-out cross-validation via `az.loo()`
    - **WAIC**: Widely applicable information criterion via `az.waic()`
    - **Model Comparison**: Compare multiple models with `az.compare()`
    - **Outlier Detection**: Identify poorly fit observations
    
    Each MCMC draw × observation gets a log-likelihood value, representing
    how well that parameter draw explains that specific observation.
    
    Examples
    --------

    ```python
    from brmspy import fit
    import arviz as az
    
    # Fit model (log_lik included automatically)
    result = fit("y ~ x", data={"y": [1, 2, 3], "x": [1, 2, 3]})
    
    # Model comparison with LOO-CV
    loo_result = az.loo(result.idata)
    print(loo_result)
    
    # Compare multiple models
    model1_idata = fit("y ~ x", data=data1).idata
    model2_idata = fit("y ~ x + x2", data=data2).idata
    comparison = az.compare({"model1": model1_idata, "model2": model2_idata})
    ```

    See Also
    --------
    brmspy.brms.log_lik : High-level wrapper that calls this
    generic_pred_to_idata : Generic conversion function
    arviz.loo : Leave-one-out cross-validation
    arviz.waic : WAIC computation
    arviz.compare : Model comparison
    """
    return generic_pred_to_idata(r_log_lik_obj, brmsfit_obj, newdata=newdata, var_name=var_name, az_name="log_likelihood")




from collections.abc import Mapping, Sequence

def py_to_r(obj):
    """
    Convert arbitrary Python objects to R objects via rpy2.
    
    Comprehensive converter that handles nested structures (dicts, lists),
    DataFrames, arrays, and scalars. Uses rpy2's converters with special
    handling for dictionaries (→ R named lists) and lists of dicts.
    
    Parameters
    ----------
    obj : any
        Python object to convert. Supported types:
        - None → R NULL
        - dict → R named list (ListVector), recursively
        - list/tuple of dicts → R list of named lists
        - list/tuple (other) → R vector or list
        - pd.DataFrame → R data.frame
        - np.ndarray → R vector/matrix
        - scalars (int, float, str, bool) → R atomic types
    
    Returns
    -------
    rpy2 R object
        R representation of the Python object
    
    Notes
    -----
    **Conversion Rules:**
    
    1. **None**: → R NULL
    2. **DataFrames**: → R data.frame (via pandas2ri)
    3. **Dictionaries**: → R named list (ListVector), recursively converting values
    4. **Lists of dicts**: → R list with 1-based indexed names containing named lists
    5. **Other lists/tuples**: → R vectors or lists (via rpy2 default)
    6. **NumPy arrays**: → R vectors/matrices (via numpy2ri)
    7. **Scalars**: → R atomic values
    
    **Recursive Conversion:**
    
    Dictionary values are recursively converted, allowing nested structures:
    ```python
    {'a': {'b': [1, 2, 3]}}  →  list(a = list(b = c(1, 2, 3)))
    ```
    
    **List of Dicts:**
    
    Lists containing only dicts are converted to R lists with 1-based indexing:
    ```python
    [{'x': 1}, {'x': 2}]  →  list("1" = list(x = 1), "2" = list(x = 2))
    ```
    
    Examples
    --------

    ```python
    from brmspy.helpers.conversion import py_to_r
    import numpy as np
    import pandas as pd
    
    # Scalars
    py_to_r(5)        # R: 5
    py_to_r("hello")  # R: "hello"
    py_to_r(None)     # R: NULL
    
    # Arrays
    py_to_r(np.array([1, 2, 3]))  # R: c(1, 2, 3)
    
    # DataFrames
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    py_to_r(df)  # R: data.frame(x = c(1, 2), y = c(3, 4))
    ```

    See Also
    --------
    r_to_py : Convert R objects back to Python
    kwargs_r : Convert keyword arguments dict for R function calls
    brmspy.brms.fit : Uses this for converting data to R
    """
    if isinstance(obj, RListVectorExtension):
        return obj.r
        
    if obj is None:
        return ro.NULL

    if isinstance(obj, ro.Sexp):
        return obj

    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        

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
    Convert R objects to Python objects via rpy2.
    
    Comprehensive converter that handles R lists (named/unnamed), vectors,
    formulas, and language objects. Provides sensible Python equivalents
    for all R types with special handling for edge cases.
    
    Parameters
    ----------
    obj : rpy2 R object
        R object to convert to Python
    
    Returns
    -------
    any
        Python representation of the R object:
        - R NULL → None
        - Named list → dict (recursively)
        - Unnamed list → list (recursively)
        - Length-1 vector → scalar (int, float, str, bool)
        - Length-N vector → list of scalars
        - Formula/Language object → str (descriptive representation)
        - Other objects → default rpy2 conversion or str fallback
    
    Notes
    -----
    **Conversion Rules:**
    
    1. **R NULL**: → Python None
    2. **Atomic vectors** (numeric, character, logical):
       - Length 1: → Python scalar (int, float, str, bool)
       - Length >1: → Python list of scalars
    3. **Named lists** (ListVector with names): → Python dict, recursively
    4. **Unnamed lists**: → Python list, recursively
    5. **Formulas** (e.g., `y ~ x`): → String representation
    6. **Language objects** (calls, expressions): → String representation
    7. **Functions**: → String representation
    8. **Everything else**: Try default rpy2 conversion, fallback to string
    
    **Recursive Conversion:**
    
    List elements and dictionary values are recursively converted:
    ```R
    list(a = list(b = c(1, 2)))  →  {'a': {'b': [1, 2]}}
    ```
    
    **Safe Fallback:**
    
    R language objects, formulas, and functions are converted to descriptive
    strings rather than attempting complex conversions that might fail.
    
    Examples
    --------

    ```python
    from brmspy.helpers.conversion import r_to_py
    import rpy2.robjects as ro
    
    # R NULL
    r_to_py(ro.NULL)  # None
    
    # Scalars
    r_to_py(ro.IntVector([5]))    # 5
    r_to_py(ro.FloatVector([3.14]))  # 3.14
    r_to_py(ro.StrVector(["hello"]))  # "hello"
    
    # Vectors
    r_to_py(ro.IntVector([1, 2, 3]))  # [1, 2, 3]
    ```
    
    See Also
    --------
    py_to_r : Convert Python objects to R
    brmspy.brms.summary : Returns Python-friendly summary dict
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


