import os
from pathlib import Path
from typing import Callable, List, Optional, Union, cast
import rpy2.robjects as ro
from packaging.version import Version

from brmspy.helpers.log import log, log_warning


def _try_force_unload_package(package: str, uninstall = True) -> None:
    """
    Try to unload and remove an R package as aggressively as possible.

    - Logs each step (start / ok / error) from inside R.
    - Does NOT raise on R-side failures; only logs them.
    - May still fail on Windows if DLLs are locked or dependencies keep it loaded.
    """
    r_code = f"""
      pkg <- "{package}"
      uninstall <- "{uninstall}"

      log_step <- function(step, expr) {{
        msg_prefix <- paste0("[unload:", pkg, "][", step, "] ")
        cat(msg_prefix, "start\\n", sep = "")
        res <- try(eval(expr), silent = TRUE)

        if (inherits(res, "try-error")) {{
          cond <- attr(res, "condition")
          if (!is.null(cond)) {{
            cat(msg_prefix, "ERROR: ", conditionMessage(cond), "\\n", sep = "")
          }} else {{
            cat(msg_prefix, "ERROR: ", as.character(res), "\\n", sep = "")
          }}
          FALSE
        }} else {{
          cat(msg_prefix, "ok\\n", sep = "")
          TRUE
        }}
      }}

      .unload_pkg <- function(pkg) {{
        

        # 1) Detach from search path
        log_step("detach_search", quote({{
          search_name <- paste0("package:", pkg)
          if (search_name %in% search()) {{
            detach(search_name, unload = TRUE, character.only = TRUE)
          }}
        }}))

        # 2) Unload namespace
        log_step("unloadNamespace", quote({{
          if (pkg %in% loadedNamespaces()) {{
            unloadNamespace(pkg)
          }}
        }}))

        # 3) pkgload (devtools-style unload)
        log_step("pkgload::unload", quote({{
          if (requireNamespace("pkgload", quietly = TRUE)) {{
            pkgload::unload(pkg)
          }}
        }}))

        # 4) DLL unload if still registered
        log_step("library.dynam.unload", quote({{
          dlls <- getLoadedDLLs()
          if (pkg %in% rownames(dlls)) {{
            dll_info <- dlls[[pkg]]
            dll_name <- dll_info[["name"]]
            libpath  <- dirname(dll_info[["path"]])
            library.dynam.unload(chname = dll_name,
                                 package = pkg,
                                 libpath = libpath)
          }}
        }}))

        # 5) Remove package from library if still installed
        if (uninstall == "True") {{
        log_step("remove.packages", quote({{
          ip <- installed.packages()
          if (pkg %in% rownames(ip)) {{
            remove.packages(pkg)
          }}
        }}))
        }}
      }}

      .unload_pkg(pkg)
    """

    try:
        log(f"Attempting aggressive unload of R package '{package}'")
        ro.r(r_code)
        log(f"Aggressive unload completed for '{package}'")
    except Exception as e:
        # rpy2 / transport-level failure – log, but don't kill caller
        log_warning(f"Aggressive unload of '{package}' raised a Python/rpy2 exception: \n{e}")
    


def _forward_github_token_to_r() -> None:
    """Forward GITHUB_PAT / GITHUB_TOKEN from Python env to R's Sys.getenv."""
    try:
        kwargs = {}
        pat = os.environ.get("GITHUB_PAT")
        token = os.environ.get("GITHUB_TOKEN")

        if not pat and not token:
            return
        
        r_setenv = cast(Callable, ro.r["Sys.setenv"])

        if pat:
            kwargs["GITHUB_PAT"] = pat
        elif token:
            kwargs["GITHUB_TOKEN"] = token

        if kwargs:
            r_setenv(**kwargs)
    except Exception as e:
        log_warning(f"{e}")
        return

def _get_r_pkg_version(package: str) -> Optional[Version]:
    """
    Get installed R package version without loading the package.
    """
    try:
        # 1. packageDescription reads the DESCRIPTION file from disk.
        # 2. fields='Version' extracts just the version string.
        # 3. We wrap it in a check: if the package is missing, packageDescription 
        #    returns NA (and warns). We force an error with stop() if it is NA
        #    so the Python 'except' block catches it correctly.
        expr = f"""
        v <- utils::packageDescription('{package}', fields = 'Version')
        if (is.na(v)) stop('Package not found')
        v
        """
        
        # ro.r returns a vector; [0] gets the string value
        v_str =cast(List, ro.r(expr))[0]
        return Version(v_str)
        
    except Exception:
        return None

def _get_r_pkg_installed(package: str,
                         lib_loc: Optional[Union[str, Path]] = None) -> bool:
    """
    Return True if `package` is installed in the current R library paths,
    without loading the package/namespace.

    Parameters
    ----------
    package :
        R package name.
    lib_loc :
        Optional library path to restrict the search to. If None, uses
        whatever `.libPaths()` is currently set to.
    """
    from rpy2.robjects.packages import isinstalled

    if lib_loc is not None:
        lib_loc = str(lib_loc)

    try:
        # rpy2.robjects.packages.isinstalled() already returns a Python bool
        return isinstalled(package, lib_loc=lib_loc)
    except Exception:
        # Fail closed rather than blowing up the whole session
        return False


def _r_namespace_loaded(pkg: str) -> bool:
    """
    Return True if `pkg`'s namespace is loaded in this R session.
    """
    expr = f'"{pkg}" %in% loadedNamespaces()'
    res = cast(List, ro.r(expr))
    # res is an R logical vector; res[0] is a logical scalar, not a "TRUE"/"FALSE" string
    return str(res[0]).lower().strip() == "true"


def _r_package_attached(pkg: str) -> bool:
    expr = f'paste0("package:", "{pkg}") %in% search()'
    res = cast(List, ro.r(expr))
    return str(res[0]).lower().strip() == "true"