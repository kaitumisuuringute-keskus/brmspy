"""
Surface for the `brms.manage()` context.

This module is safe to import in the main process (no top-level rpy2 imports).
All heavy work is executed in the worker process when these methods are called
via the session context proxy (ClassProxy).

Only names listed in __all__ are exposed by the manage() context proxy.
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["ManageModule"]

# Planned (not implemented yet in brmspy runtime layer; keep as TODO comments only):
# - install_runtime
# - uninstall_rpackage
# - list_rpackages
# - import_rpackages


class ManageModule:
    @staticmethod
    def install_rpackage(
        name: str,
        version: str | None = None,
        repos_extra: list[str] | None = None,
    ) -> None:
        """
        Install an R package into the active environment/user library.

        Executed in worker.
        """
        from brmspy._runtime._r_packages import install_package

        return install_package(name, version=version, repos_extra=repos_extra)

    @staticmethod
    def install_brms(
        *,
        use_prebuilt: bool = False,
        install_rtools: bool = False,
        brms_version: str | None = None,
        cmdstanr_version: str | None = None,
        install_rstan: bool = True,
        install_cmdstanr: bool = True,
        rstan_version: str | None = None,
        activate: bool = True,
    ) -> Path | None:
        """
        Install brms and toolchain dependencies.

        Executed in worker.
        """
        from brmspy._runtime import install_brms as _install_brms

        return _install_brms(
            use_prebuilt=use_prebuilt,
            install_rtools=install_rtools,
            brms_version=brms_version,
            cmdstanr_version=cmdstanr_version,
            install_rstan=install_rstan,
            install_cmdstanr=install_cmdstanr,
            rstan_version=rstan_version,
            activate=activate,
        )
