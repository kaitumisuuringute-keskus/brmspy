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
# - list_rpackages
# - import_rpackages


class ManageModule:
    @staticmethod
    def install_runtime(*, install_rtools: bool = False):
        from brmspy._runtime import install_runtime as _install_runtime

        return _install_runtime(install_rtools=install_rtools)

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
    def uninstall_rpackage(name: str) -> bool:
        from brmspy._runtime._r_packages import remove_package

        return remove_package(name)

    @staticmethod
    def import_rpackages(*names: str):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as ro

        for name in names:
            importr(name)

    @staticmethod
    def is_rpackage_loaded(name: str) -> bool:
        from brmspy._runtime._r_env import is_namespace_loaded, is_package_attached

        return is_namespace_loaded(name) or is_package_attached(name)

    @staticmethod
    def get_rpackage_version(name: str) -> str | None:
        from brmspy._runtime._r_packages import get_package_version

        return get_package_version(name)

    @staticmethod
    def is_rpackage_installed(name: str) -> bool:
        from brmspy._runtime._r_packages import get_package_version

        return get_package_version(name) is not None

    @staticmethod
    def _unload_rpackage(name: str) -> bool:
        from brmspy._runtime._r_env import unload_package

        return unload_package(name)

    @staticmethod
    def get_lib_paths() -> list[str]:
        from brmspy._runtime._r_env import get_lib_paths

        return get_lib_paths()

    @staticmethod
    def get_cmdstan_path() -> str | None:
        from brmspy._runtime._r_env import get_cmdstan_path

        return get_cmdstan_path()
