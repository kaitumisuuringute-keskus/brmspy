import rpy2.robjects.packages as rpackages

_brms = None
_cmdstanr = None
_rstan = None
_posterior = None
_base = None

def _get_brms():
    """
    Lazy import of brms R package.
    
    Returns
    -------
    brms module
        Imported brms R package
    
    Raises
    ------
    ImportError
        If brms is not installed
    """
    global _brms, _cmdstanr, _base, _posterior, _rstan
    if _brms is None:
        print("brmspy: Importing R libraries...")
        try:
            try:
                _cmdstanr = rpackages.importr("cmdstanr")
            except Exception as e:
                _cmdstanr = None
            try:
                _rstan = rpackages.importr("rstan")
            except Exception as e:
                _rstan = None
            _posterior = rpackages.importr("posterior")
            _brms = rpackages.importr("brms")
            _base = rpackages.importr("base")
            print("brmspy: R libraries imported!")
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

def _get_base():
    global _base
    return _base

def _invalidate_singletons():
    global _brms, _cmdstanr, _base, _posterior
    _brms = None
    _cmdstanr = None
    _base = None
    _posterior = None