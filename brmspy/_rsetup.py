

def _check_rpaths() -> None:
    import os

    if not os.environ.get("R_HOME"):
        print(
            "[brmspy][CRITICAL WARNING] R_HOME env var not found! The environment may not work!\n" \
            "Set using `export R_HOME=$(R RHOME)`"
        )
    if not os.environ.get("LD_LIBRARY_PATH"):
        print(
            "[brmspy][CRITICAL WARNING] LD_LIBRARY_PATH env var not found! The environment may not work!\n" \
            'Set using `export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH}"`'
        )
    
        



def _initialise_r_safe() -> None:
    """
    Configure R for safer embedded execution.

    - Try to enforce rpy2 ABI mode (must be set before importing rpy2)
    - Disable fork-based R parallelism (future::multicore, mclapply)
    - Use future::plan(sequential) if future is available
    - Leave cmdstanr multi-core sampling alone
    """
    _check_rpaths()

    import os
    import sys

    # CFFI MODE
    if "rpy2" in sys.modules:
        if os.environ.get('RPY2_CFFI_MODE') != "ABI":
            print(
                "[brmspy][WARNING] rpy2 was imported before brmspy; cannot enforce "
                "RPY2_CFFI_MODE (env var). API and BOTH mode are known to cause "
                "instability, ABI is recommended."
            )
    elif os.environ.get('RPY2_CFFI_MODE') in ('BOTH', 'API'):
        print(
            "[brmspy][WARNING] RPY2_CFFI_MODE (env var) is set to API/BOTH. "
            "These modes are known to cause instability and segfaults; "
            "ABI is recommended."
        )
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    # THREAD SAFETY
    # Could also lead to undefined behaviour if >1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


    import rpy2.robjects as ro

    ro.r(
        r"""
        # Disable fork-based mechanisms that are unsafe in embedded R
        options(
          mc.cores = 1L,             # parallel::mclapply -> serial
          future.fork.enable = FALSE, # disable future::multicore
          loo.cores = 1L # deprecated but still respected, for now.
        )

        # If 'future' is installed, force sequential backend
        if (requireNamespace("future", quietly = TRUE)) {
          future::plan(future::sequential)
        }
        """
    )