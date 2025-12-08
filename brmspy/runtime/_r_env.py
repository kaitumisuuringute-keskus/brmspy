"""
R environment operations: libPaths, cmdstan path, package loading.
Each function does exactly one R operation. Stateless.
"""

import os
from pathlib import Path
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
        result = cast(ro.ListVector, ro.r("suppressWarnings(suppressMessages(cmdstanr::cmdstan_path()))"))
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



def unload_package(name: str) -> bool:
    """
    Also known as footgun. Don't call without very good reason.

    Attempt to unload package. Returns True if successful.
    Tries: detach -> unloadNamespace -> library.dynam.unload
    Does NOT uninstall.
    """

    detach_only = False

    r_code = f"""
      pkg <- "{name}"
      detach_only <- {str(detach_only).upper()}
      
      .unload_pkg <- function(pkg, detach_only) {{
        success <- TRUE
        
        # Always try to detach from search path first
        tryCatch({{
          search_name <- paste0("package:", pkg)
          if (search_name %in% search()) {{
            detach(search_name,
                   unload = !detach_only,
                   character.only = TRUE)
          }}
        }}, error = function(e) {{ success <<- FALSE }})

        if (detach_only) {{
          # do *not* touch namespace or DLL
          return(success)
        }}

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
            library.dynam.unload(
              chname  = dll_name,
              package = pkg,
              libpath = libpath
            )
          }}
        }}, error = function(e) {{}})

        success
      }}
      
      .unload_pkg(pkg, detach_only)
    """
    
    try:
        result = cast(List, ro.r(r_code))
        return str(result[0]).lower().strip() == "true"
    except Exception:
        return False



def _find_libpath_packages(libpath: Path | None, include_not_loaded: bool = False) -> List[str]:
    import rpy2.robjects as ro
    if libpath is None:
        return []
    pkgs = cast(Callable, ro.r("""
function(runtime_root, include_not_loaded = FALSE) {
  lib_root <- file.path(runtime_root, "Rlib")
  lib_root <- normalizePath(lib_root, winslash = "/", mustWork = TRUE)

  attached <- sub("^package:", "", grep("^package:", search(), value = TRUE))
  ns <- loadedNamespaces()
  pkgs <- unique(c(attached, ns))
                               
  # optionally add packages that are installed in this lib_root
  if (isTRUE(include_not_loaded)) {
    inst <- tryCatch(
      installed.packages(lib.loc = lib_root)[, "Package"],
      error = function(e) character(0L)
    )
    pkgs <- unique(c(pkgs, inst))
  }

  res <- vapply(pkgs, function(p) {
    path <- suppressWarnings(tryCatch(
      find.package(p, quiet = TRUE)[1],
      error = function(e) NA_character_
    ))

    if (is.na(path) || !nzchar(path)) {
      return(NA_character_)
    }

    path <- normalizePath(path, winslash = "/", mustWork = FALSE)

    if (startsWith(path, lib_root)) p else NA_character_
  }, character(1L))

  res[!is.na(res)]
}
    """))(libpath.as_posix(), include_not_loaded)

    pkgs = [str(v) for v in cast(ro.StrVector, pkgs)]

    return pkgs


def _compute_unload_order(pkgs: List[str] | None) -> List[str] | None:
    if pkgs is None:
        return None
    if len(pkgs) == 0:
        return []
    import rpy2.robjects as ro
    fun = cast(Callable, ro.r("""
function(pkgs) {
  pkgs <- unique(as.character(pkgs))
  if (!length(pkgs)) return(character(0L))

  # Try to get metadata; if this fails, just treat as "no deps info"
  ip <- tryCatch(
    installed.packages(lib.loc = .libPaths(), noCache = TRUE)[, "Package"],
    error = function(e) character(0L)
  )

  pkgs_with_meta <- intersect(pkgs, ip)

  deps <- vector("list", length(pkgs))
  names(deps) <- pkgs

  if (length(pkgs_with_meta)) {
    d <- tools::package_dependencies(
      pkgs_with_meta,
      which     = c("Depends", "Imports", "LinkingTo"),
      recursive = FALSE,
      reverse   = TRUE
    )
    # Keep only edges within our original pkgs set
    d <- lapply(d, function(x) intersect(x, pkgs))
    deps[names(d)] <- d
  }

  remaining <- pkgs
  order <- character(0L)

  while (length(remaining)) {
    has_dependents <- unique(unlist(deps[remaining]))
    leaves <- setdiff(remaining, has_dependents)

    if (!length(leaves)) {
      # Cycle or no dep info: just append the rest
      order <- c(order, remaining)
      break
    }

    order <- c(order, leaves)
    remaining <- setdiff(remaining, leaves)
  }

  order
}
"""))
    lv = cast(ro.StrVector, fun(ro.StrVector(pkgs)))
    return [str(v) for v in lv]

def _unload_libpath_packages(libpath: Path | None) -> None:
    if not libpath or not libpath.exists():
        return
    pkgs = _find_libpath_packages(libpath)
    if len(pkgs) == 0:
        return
    pkgs = _compute_unload_order(pkgs)
    if not pkgs:
        return
    for pkg in pkgs:
        try:
            unload_package(pkg)
        except Exception as e:
            log_warning(f"{e}")
    
    

# === Mutations ===

def set_lib_paths(paths: list[str]) -> None:
    """Set .libPaths() in R."""
    
    current = [str(p) for p in cast(ro.StrVector, ro.r(".libPaths()"))]
    current = [p for p in current if ".brmspy/runtime/" not in p and ".brmspy\\runtime\\" not in p]
    new_paths = list(dict.fromkeys(list(paths) + current))
    r_fun = cast(Callable, ro.r('.libPaths'))
    r_fun(ro.StrVector(new_paths))


def set_cmdstan_path(path: str | None) -> None:
    """Set cmdstanr::set_cmdstan_path()."""
    import logging
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