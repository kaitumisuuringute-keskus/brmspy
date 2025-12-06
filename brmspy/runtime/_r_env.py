"""
R environment operations: libPaths, cmdstan path, package loading.
Each function does exactly one R operation. Stateless.
"""

import os
from typing import Callable, List, cast
import rpy2.robjects as ro

from brmspy.helpers.log import log_warning


# === Queries ===

def get_lib_paths() -> list[str]:
    """Get current .libPaths() from R."""
    result = cast(ro.ListVector, ro.r('.libPaths()'))
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
    Attempt to unload package. Returns True if successful.
    Tries: detach -> unloadNamespace -> library.dynam.unload
    Does NOT uninstall.
    """
    r_code = f"""
      pkg <- "{name}"
      
      .unload_pkg <- function(pkg) {{
        success <- TRUE
        
        # 1) Detach from search path
        tryCatch({{
          search_name <- paste0("package:", pkg)
          if (search_name %in% search()) {{
            detach(search_name, unload = TRUE, character.only = TRUE)
          }}
        }}, error = function(e) {{ success <<- FALSE }})
        
        # 2) Unload namespace
        tryCatch({{
          if (pkg %in% loadedNamespaces()) {{
            unloadNamespace(pkg)
          }}
        }}, error = function(e) {{ success <<- FALSE }})
        
        # 3) pkgload (devtools-style unload)
        tryCatch({{
          if (requireNamespace("pkgload", quietly = TRUE)) {{
            pkgload::unload(pkg)
          }}
        }}, error = function(e) {{}})
        
        # 4) DLL unload if still registered
        tryCatch({{
          dlls <- getLoadedDLLs()
          if (pkg %in% rownames(dlls)) {{
            dll_info <- dlls[[pkg]]
            dll_name <- dll_info[["name"]]
            libpath  <- dirname(dll_info[["path"]])
            library.dynam.unload(chname = dll_name,
                                package = pkg,
                                libpath = libpath)
          }}
        }}, error = function(e) {{}})
        
        return(success)
      }}
      
      .unload_pkg(pkg)
    """
    
    try:
        result = cast(List, ro.r(r_code))
        return str(result[0]).lower().strip() == "true"
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