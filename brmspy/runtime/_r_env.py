"""
R environment operations: libPaths, cmdstan path, package loading.
Each function does exactly one R operation. Stateless.
"""

import os
from typing import Callable, List, cast
import rpy2.robjects as ro

from brmspy.helpers.log import log_warning


# Core R packages that should never be unloaded
BASE_PACKAGES = frozenset({
    "base", "compiler", "datasets", "grDevices", "graphics",
    "grid", "methods", "parallel", "splines", "stats",
    "stats4", "tcltk", "tools", "utils", "translations"
})

# === Queries ===

def get_lib_paths() -> list[str]:
    """Get current .libPaths() from R."""
    result = cast(ro.ListVector, ro.r('.libPaths()'))
    return [str(p) for p in result]

def get_base_packages() -> frozenset[str]:
    """
    Get the set of base R packages that should never be unloaded.
    
    Returns a frozen set of package names including base, stats, utils, etc.
    """
    return BASE_PACKAGES


def get_loaded_namespaces() -> list[str]:
    """Get list of currently loaded R namespaces."""
    result = cast(ro.ListVector, ro.r('loadedNamespaces()'))
    return [str(p) for p in result]


def get_attached_packages() -> list[str]:
    """Get list of packages currently on the search path."""
    result = cast(ro.ListVector, ro.r('sub("^package:", "", grep("^package:", search(), value = TRUE))'))
    return [str(p) for p in result]


def get_cmdstan_path() -> str | None:
    """Get current cmdstanr::cmdstan_path() or None."""
    try:
        result = cast(ro.ListVector, ro.r("cmdstanr::cmdstan_path()"))
        return str(result[0]) if result else None
    except Exception:
        return None


def is_namespace_loaded(name: str) -> bool:
    """Check if package namespace is loaded."""
    expr = f'"{name}" %in% loadedNamespaces()'
    res = cast(ro. ListVector, ro.r(expr))
    return str(res[0]).lower().strip() == "true"


def is_package_attached(name: str) -> bool:
    """Check if package is on search path."""
    expr = f'paste0("package:", "{name}") %in% search()'
    res = cast(ro.ListVector, ro.r(expr))
    return str(res[0]).lower().strip() == "true"


# === Mutations ===

def set_lib_paths(paths: list[str]) -> None:
    """Set .libPaths() in R."""
    
    current = [str(p) for p in cast(ro.ListVector, ro.r(".libPaths()"))]
    current = [p for p in current if ".brmspy" not in p]
    new_paths = list(dict.fromkeys(list(paths) + current))
    r_fun = cast(Callable, ro.r('.libPaths'))
    r_fun(ro.StrVector(new_paths))


def set_cmdstan_path(path: str | None) -> None:
    """Set cmdstanr::set_cmdstan_path()."""
    try:
      if path is None:
          path_str = "NULL"
      else:
          path_str = f'"{path}"'
      ro.r(f'''
      if (!requireNamespace("cmdstanr", quietly = TRUE)) {{
        stop("cmdstanr is not available in rlibs")
      }}
      cmdstanr::set_cmdstan_path(path={path_str})
      ''')
    except Exception as e:
        log_warning(f"Failed to set cmdstan_path to {path}: {e}")


def unload_package(name: str) -> bool:
    """
    Attempt to unload a single R package. Returns True if successful.
    
    Performs in order:
    1. Detach from search path (if attached)
    2. Unload namespace (if loaded)
    3. Unload DLL (if loaded - critical for Windows)
    
    Each step is independent - failure of one doesn't prevent others.
    Does NOT uninstall the package.
    
    Parameters
    ----------
    name : str
        Package name to unload
        
    Returns
    -------
    bool
        True if package is no longer loaded after this call
    """
    r_code = f'''
    (function(pkg) {{
        # 1) Detach from search path
        tryCatch({{
            search_name <- paste0("package:", pkg)
            if (search_name %in% search()) {{
                detach(search_name, unload = TRUE, character.only = TRUE, force = TRUE)
            }}
        }}, error = function(e) NULL)
        
        # 2) Unload namespace  
        tryCatch({{
            if (pkg %in% loadedNamespaces()) {{
                unloadNamespace(pkg)
            }}
        }}, error = function(e) NULL)
        
        # 3) Unload DLL if still registered (critical for Windows)
        tryCatch({{
            dlls <- getLoadedDLLs()
            if (pkg %in% names(dlls)) {{
                dll_info <- dlls[[pkg]]
                # libpath should be the package install dir, not the DLL dir
                pkg_libpath <- dirname(dirname(normalizePath(dll_info[["path"]], 
                                                             winslash = "/", 
                                                             mustWork = FALSE)))
                library.dynam.unload(dll_info[["name"]], 
                                    package = pkg,
                                    libpath = pkg_libpath)
            }}
        }}, error = function(e) NULL)
        
        # Return whether package is now unloaded
        return(!(pkg %in% loadedNamespaces()))
    }})('{name}')
    '''
    
    try:
        result = cast(List, ro.r(r_code))
        return str(result[0]).upper() == "TRUE"
    except Exception:
        return False


def remove_package(name: str) -> bool:
    """
    Remove (uninstall) an R package from all library paths.
    
    OS-safe: handles Windows path issues and locked file fallbacks.
    
    Parameters
    ----------
    name : str
        Package name to remove
        
    Returns
    -------
    bool
        True if package was removed or wasn't installed
    """
    r_code = f'''
    (function(pkg) {{
        removed <- FALSE
        libs <- .libPaths()
        
        for (lib in libs) {{
            pkg_path <- file.path(lib, pkg)
            if (dir.exists(pkg_path)) {{
                tryCatch({{
                    # Use normalized path for cross-platform safety
                    lib_norm <- normalizePath(lib, winslash = "/", mustWork = FALSE)
                    suppressWarnings(remove.packages(pkg, lib = lib_norm))
                    removed <- TRUE
                }}, error = function(e) {{
                    # Windows fallback: try direct deletion if DLLs are unloaded
                    if (.Platform$OS.type == "windows") {{
                        tryCatch({{
                            unlink(pkg_path, recursive = TRUE, force = TRUE)
                            removed <- TRUE
                        }}, error = function(e2) NULL)
                    }}
                }})
            }}
        }}
        
        # Check if actually removed
        return(!dir.exists(file.path(.libPaths()[1], pkg)))
    }})('{name}')
    '''
    
    try:
        result = cast(List, ro.r(r_code))
        return str(result[0]).upper() == "TRUE"
    except Exception:
        return False


def run_gc() -> None:
    """Run garbage collection in both Python and R."""
    import gc
    gc.collect()
    try:
        ro.r('gc()')
    except Exception:
        pass



def forward_github_token() -> None:
    """Copy GITHUB_TOKEN/GITHUB_PAT to R's Sys.setenv."""
    try:
        kwargs = {}
        pat = os.environ.get("GITHUB_PAT")
        token = os.environ.get("GITHUB_TOKEN")
        
        if not pat and not token:
            return
        
        r_setenv = cast(Callable, ro.r("Sys.setenv"))
        
        if pat:
            kwargs["GITHUB_PAT"] = pat
        elif token:
            kwargs["GITHUB_TOKEN"] = token
        
        if kwargs:
            r_setenv(**kwargs)
    except Exception:
        pass