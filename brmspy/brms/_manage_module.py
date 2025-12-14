"""
Manage brmspy runtimes and R environments.

This module defines the surface returned by `brmspy.brms.manage()`.

The file is safe to import in the *main* Python process (no top-level
`rpy2.robjects` imports). In normal use these methods are invoked through the
`manage()` context, and the actual work executes in the worker process that
hosts the embedded R session.

Example:
```python
env = "mrp"
if not brms.environment_exists(env):
    with brms.manage(environment_name=env) as ctx:
        ctx.install_brms(use_prebuilt=True)
        ctx.install_rpackage("MCMCglmm")
else:
    brms.environment_activate(env)
```

Notes
-----
- Use the context manager to ensure the worker (and its embedded R session) is
  started with the desired environment configuration.
- Calling these methods directly in the main process is unsupported and may
  reintroduce the same stability issues that the worker isolation is designed
  to avoid.
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["ManageModule"]

# Planned (not implemented yet in brmspy runtime layer; keep as TODO comments only):
# - list_rpackages
# - import_rpackages


class ManageModule:
    """
    Management surface returned by `brmspy.brms.manage()`.

    The returned object is a *proxy* that executes these methods inside the
    worker process. Use it to install brms/toolchains, manage R packages in the
    active environment, and query basic runtime state.

    Notes
    -----
    The worker process must be able to run R and (depending on the installation
    mode) may require an OS toolchain for compiling packages / CmdStan.
    """

    @staticmethod
    def install_runtime(*, install_rtools: bool = False) -> Path | None:
        """
        Install the prebuilt brmspy runtime bundle.

        This is a convenience wrapper around `install_brms(use_prebuilt=True)`.
        It downloads (if necessary) a precompiled runtime and optionally activates it.

        Parameters
        ----------
        install_rtools : bool, default=False
            If ``True``, install Rtools on Windows if missing.

        Returns
        -------
        pathlib.Path or None
            Path to the installed runtime directory (prebuilt mode). Returns ``None``
            if no runtime was installed (unexpected for prebuilt mode).

        Raises
        ------
        RuntimeError
            If no compatible prebuilt runtime exists for the current platform.

        Examples
        --------
        >>> from brmspy import brms
        >>> with brms.manage(environment_name="default") as ctx:
        ...     runtime = ctx.install_runtime()
        """
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
        Install brms and its toolchain dependencies.

        In traditional mode (``use_prebuilt=False``), this installs into the active R
        library (typically the active brmspy environment) and may build CmdStan from
        source.

        In prebuilt mode (``use_prebuilt=True``), this downloads a brmspy runtime
        bundle (R packages + CmdStan) and can activate it.

        Parameters
        ----------
        use_prebuilt : bool, default=False
            If ``True``, use a prebuilt runtime bundle instead of installing via R.
        install_rtools : bool, default=False
            If ``True``, install Rtools on Windows if missing.
        brms_version : str or None, default=None
            Version spec for the brms R package (traditional mode only). ``None`` means
            "latest".
        cmdstanr_version : str or None, default=None
            Version spec for cmdstanr (traditional mode only). ``None`` means "latest".
        install_rstan : bool, default=True
            If ``True``, install rstan (traditional mode).
        install_cmdstanr : bool, default=True
            If ``True``, install cmdstanr and CmdStan (traditional mode).
        rstan_version : str or None, default=None
            Version spec for rstan (traditional mode only). ``None`` means "latest".
        activate : bool, default=True
            If ``True`` and ``use_prebuilt=True``, activate the downloaded runtime in the
            worker's embedded R session.

        Returns
        -------
        pathlib.Path or None
            If ``use_prebuilt=True``, returns the installed runtime directory.
            If ``use_prebuilt=False``, returns ``None``.

        Raises
        ------
        RuntimeError
            If installation fails (for example missing toolchain, or no compatible
            prebuilt runtime exists).

        Examples
        --------
        Prebuilt (fast) install:

        >>> from brmspy import brms
        >>> with brms.manage(environment_name="default") as ctx:
        ...     ctx.install_brms(use_prebuilt=True)

        Traditional (R installs + builds CmdStan):

        >>> from brmspy import brms
        >>> with brms.manage(environment_name="default") as ctx:
        ...     ctx.install_brms(use_prebuilt=False, install_cmdstanr=True, install_rstan=False)
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
        Install an R package into the active environment library.

        Parameters
        ----------
        name : str
            R package name (e.g. ``"MCMCglmm"``).
        version : str or None, default=None
            Optional version spec. ``None`` means "latest".
        repos_extra : list[str] or None, default=None
            Extra repositories to add (for example R-universe URLs).

        Returns
        -------
        None

        Notes
        -----
        This installs into the *active* R library path (usually the brmspy environment
        library), not into the system R library tree.

        Examples
        --------
        >>> from brmspy import brms
        >>> with brms.manage(environment_name="mrp") as ctx:
        ...     ctx.install_rpackage("MCMCglmm")
        """
        from brmspy._runtime._r_packages import install_package

        return install_package(name, version=version, repos_extra=repos_extra)

    @staticmethod
    def uninstall_rpackage(name: str) -> bool:
        """
        Uninstall an R package from the active library paths.

        Parameters
        ----------
        name : str
            R package name.

        Returns
        -------
        bool
            ``True`` if the package appears removed, otherwise ``False``.

        Notes
        -----
        Package unloading/removal can be OS-dependent (especially on Windows where DLLs
        may be locked). This function makes a best effort.

        Examples
        --------
        >>> from brmspy import brms
        >>> with brms.manage(environment_name="mrp") as ctx:
        ...     ok = ctx.uninstall_rpackage("MCMCglmm")
        """
        from brmspy._runtime._r_packages import remove_package

        return remove_package(name)

    @staticmethod
    def import_rpackages(*names: str) -> None:
        """
        Import (load) one or more R packages into the worker's embedded R session.

        This does *not* install packages. Use `install_rpackage()` first if needed.

        Parameters
        ----------
        *names : str
            One or more package names.

        Returns
        -------
        None

        Examples
        --------
        >>> from brmspy import brms
        >>> with brms.manage(environment_name="default") as ctx:
        ...     ctx.import_rpackages("brms", "cmdstanr")
        """
        from rpy2.robjects.packages import importr

        for name in names:
            importr(name)

    @staticmethod
    def is_rpackage_loaded(name: str) -> bool:
        """
        Check whether an R package is loaded in the current R session.

        Parameters
        ----------
        name : str
            R package name.

        Returns
        -------
        bool
            ``True`` if the package is loaded (namespace loaded or attached).
        """
        from brmspy._runtime._r_env import is_namespace_loaded, is_package_attached

        return is_namespace_loaded(name) or is_package_attached(name)

    @staticmethod
    def get_rpackage_version(name: str) -> str | None:
        """
        Get installed version of an R package.

        Parameters
        ----------
        name : str
            R package name.

        Returns
        -------
        str or None
            Installed version string, or ``None`` if not installed / not found.
        """
        from brmspy._runtime._r_packages import get_package_version

        return get_package_version(name)

    @staticmethod
    def is_rpackage_installed(name: str) -> bool:
        """
        Check whether an R package is installed in the active library paths.

        Parameters
        ----------
        name : str
            R package name.

        Returns
        -------
        bool
            ``True`` if installed, otherwise ``False``.
        """
        from brmspy._runtime._r_packages import get_package_version

        return get_package_version(name) is not None

    @staticmethod
    def _unload_rpackage(name: str) -> bool:
        """
        Attempt to unload an R package from the current session (advanced).

        This is intentionally private: unloading packages at runtime can be fragile.

        Parameters
        ----------
        name : str
            R package name.

        Returns
        -------
        bool
            ``True`` if the unload attempt was reported as successful.
        """
        from brmspy._runtime._r_env import unload_package

        return unload_package(name)

    @staticmethod
    def get_lib_paths() -> list[str]:
        """
        Get the current R ``.libPaths()`` search paths.

        Returns
        -------
        list[str]
            R library search paths (highest priority first).
        """
        from brmspy._runtime._r_env import get_lib_paths

        return get_lib_paths()

    @staticmethod
    def get_cmdstan_path() -> str | None:
        """
        Get the current CmdStan path configured in cmdstanr.

        Returns
        -------
        str or None
            CmdStan directory path, or ``None`` if not configured / cmdstanr unavailable.
        """
        from brmspy._runtime._r_env import get_cmdstan_path

        return get_cmdstan_path()
