

def _initialise_rpaths() -> None:
    import os, subprocess
    from pathlib import Path
    import platform

    # ensure R_HOME
    if not os.environ.get("R_HOME"):
        try:
            r_home = subprocess.check_output(
                ["R", "RHOME"], text=True, timeout=30
            ).strip()
            os.environ["R_HOME"] = r_home
        except Exception:
            return # Let rpy2 fail naturally later if R isn't found

    r_home = Path(os.environ["R_HOME"])
    system = platform.system()


    # ensure LD_LIBRARY_PATH and link
    
    # 3. FORCE LINKING (The "methods.so" Fix)
    # We explicitly load libR into the global process memory. 
    # This solves the "libR.so not found" error without relying on LD_LIBRARY_PATH race conditions.
    
    lib_path = None
    if system == "Linux":
        lib_path = r_home / "lib" / "libR.so"
    elif system == "Darwin":
        lib_path = r_home / "lib" / "libR.dylib"
    
    # Linux/Mac: Pre-load with RTLD_GLOBAL
    if lib_path and lib_path.exists():
        import ctypes
        try:
            # RTLD_GLOBAL (0x100) makes symbols available to subsequently loaded libraries (like methods.so)
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            # print(f"[brmspy] Pre-linked R library: {lib_path}")
        except OSError:
            # If this fails, we just hope the environment is already correct
            pass
    


def _initialise_r_safe() -> None:
    """
    Configure R for safer embedded execution.

    - Try to enforce rpy2 ABI mode (must be set before importing rpy2)
    - Disable fork-based R parallelism (future::multicore, mclapply)
    - Use future::plan(sequential) if future is available
    - Leave cmdstanr multi-core sampling alone
    """
    _initialise_rpaths()

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