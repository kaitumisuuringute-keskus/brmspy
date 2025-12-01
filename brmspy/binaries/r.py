import rpy2.robjects as ro



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
        print(f"[brmspy] Attempting aggressive unload of R package '{package}'")
        ro.r(r_code)
        print(f"[brmspy] Aggressive unload completed for '{package}'")
    except Exception as e:
        # rpy2 / transport-level failure – log, but don't kill caller
        print(f"[brmspy] Aggressive unload of '{package}' raised a Python/rpy2 exception: \n{e}")
    