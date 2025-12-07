

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
    lib_dir = r_home / "lib"
    system = platform.system()


    # 2. Update LD_LIBRARY_PATH (For child processes)
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(lib_dir) not in current_ld:
        import ctypes
        new_ld = f"{lib_dir}:{current_ld}" if current_ld else str(lib_dir)
        os.environ["LD_LIBRARY_PATH"] = new_ld

        # 3. FORCE LINKING (The Nuclear Option)
        # We explicitly load ALL shared objects in R_HOME/lib into global memory.
        if system in ("Linux", "Darwin") and lib_dir.exists():
            
            # Define extension based on OS
            ext = ".dylib" if system == "Darwin" else ".so"
            
            # PRIORITIZE libR! It must be loaded first or others might fail.
            core_lib = lib_dir / ("libR" + ext)
            if core_lib.exists():
                try:
                    ctypes.CDLL(str(core_lib), mode=ctypes.RTLD_GLOBAL)
                    print(f"DEBUG: Pre-loaded Core: {core_lib.name}")
                except OSError:
                    pass

            # Now load everything else in that specific folder (BLAS, LAPACK, etc.)
            # strictly non-recursive to avoid touching R packages.
            for shared_lib in lib_dir.glob(f"*{ext}"):
                # Skip libR if we already loaded it
                if shared_lib.name == core_lib.name:
                    continue
                    
                try:
                    # RTLD_GLOBAL makes these symbols available to R packages later
                    ctypes.CDLL(str(shared_lib), mode=ctypes.RTLD_GLOBAL)
                    print(f"DEBUG: Pre-loaded Dep: {shared_lib.name}")
                except OSError:
                    # Some libs might fail if they have circular deps, ignore them.
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