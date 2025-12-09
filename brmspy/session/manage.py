from contextlib import contextmanager
import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Generator, Iterator, List, Optional, Callable, Any, Union, cast
from pathlib import Path

from brmspy import brms
from brmspy._runtime import get_active_runtime
from .environment import get_environment_config
from .types import EnvironmentConfig

from .module_session import RModuleSession

@dataclass
class EnvContext:
    """Narrow surface area for R-env mutations."""
    session: RModuleSession
    _has_imported: bool = False

    # Runtime
    def install_runtime(self):
        ...

    def list_environments(self):
        ...

    def activate_environment(self, name: Optional[str]):
        ...

    def deactivate_environment(self):
        ...

    def get_active_environment(self):
        ...

    def create_environment(self, name: str) -> Path:
        ...

    # rpackages
    def install_rpackage(self, name: str, version: Optional[str] = None, repos_extra: Optional[List[str]] = None) -> None:
        result = self.session._call_remote("mod:brmspy._runtime._r_packages.install_package", name, version=version, repos_extra=repos_extra)
        return result

    def uninstall_rpackage(self, name: str):
        ...

    def list_rpackages(self) -> List[str]:
        ...

    def import_rpackages(self, *packages: List[str]):
        ...

    
    # general entrypoint
    def install_brms(
        self,
        *,
        use_prebuilt: bool = False,
        install_rtools: bool = False,
        brms_version: str | None = None,
        cmdstanr_version: str | None = None,
        install_rstan: bool = True,
        install_cmdstanr: bool = True,
        rstan_version: str | None = None,
        activate: bool = True,
        **kwargs
    ) -> Path | None:
        """
        Install brms R package, optionally cmdstanr and CmdStan compiler, or rstan.
        
        Parameters
        ----------
        brms_version : str, default="latest"
            brms version: "latest", "2.23.0", or ">= 2.20.0"
        repo : str | None, default=None
            Extra CRAN repository URL
        install_cmdstanr : bool, default=True
            Whether to install cmdstanr and build CmdStan compiler
        install_rstan : bool, default=False
            Whether to install rstan (alternative to cmdstanr)
        cmdstanr_version : str, default="latest"
            cmdstanr version: "latest", "0.8.1", or ">= 0.8.0"
        rstan_version : str, default="latest"
            rstan version: "latest", "2.32.6", or ">= 2.32.0"
        use_prebuilt: bool, default=False
            Uses fully prebuilt binaries for cmdstanr and brms and their dependencies. 
            Ignores system R libraries and uses the latest brms and cmdstanr available 
            for your system. Requires R>=4 and might not be compatible with some older
            systems or missing toolchains. Can reduce setup time by 50x.
        install_rtools: bool, default=False
            Installs RTools (windows only) if they cant be found. 
            WARNING: Modifies system path and runs the full rtools installer. 
            Use with caution!
        
        Examples
        --------
        Basic installation:
        
        ```python
        from brmspy import brms
        with brms.manage() as ctx:
            ctx.install_brms()
        ```
        Install specific version:
        
        ```python
        with brms.manage() as ctx:
            ctx.install_brms(brms_version="2.23.0")
        ```

        Use rstan instead of cmdstanr:

        ```python
        with brms.manage() as ctx:
            ctx.install_brms(install_cmdstanr=False, install_rstan=True)
        ```

        Fast installation with prebuilt binaries:
        ```python
        with brms.manage() as ctx:
            ctx.install_brms(use_prebuilt=True)
        """

        result = self.session._call_remote(
            "mod:brmspy._runtime.install_brms",
            use_prebuilt=use_prebuilt,
            install_rtools=install_rtools,
            brms_version=brms_version,
            cmdstanr_version=cmdstanr_version,
            install_rstan=install_rstan,
            install_cmdstanr=install_cmdstanr,
            rstan_version=rstan_version,
            activate=activate,
            **kwargs
        )

        active_runtime = self.session.get_active_runtime()
        if active_runtime:
            self.session._environment_conf.runtime_path = active_runtime.as_posix()

        return result



def _get_session() -> RModuleSession:
    # In main process, brms is the proxy; in worker it’s the real module.
    # This should only be used in main.
    from brmspy import brms as _brms  # or just capture outer name
    if not _brms._is_main_process:
        raise RuntimeError("environment() is only valid in the main process.")
    return cast(RModuleSession, _brms)


@contextmanager
def manage(
    *,
    environment_config: Optional[Union[EnvironmentConfig, Dict[str, str]]] = None,
    environment_name: Optional[str] = None
) -> Iterator[EnvContext]:
    """
    Run a block in a fresh R session.

    - Takes a snapshot of current runtime_conf.
    - Optionally adds a temporary R_LIBS_USER.
    - Restarts the worker before yielding.
    - Always restarts to original config on exit.
    """
    session = _get_session()

    if environment_name and environment_config:
        raise Exception("Only provide one: environment name or environment config")

    if not environment_name and environment_config:
        overrides = EnvironmentConfig.from_obj(environment_config)
    elif environment_name:
        overrides = get_environment_config(environment_name)
    else:
        overrides = None

    old_conf = session._environment_conf

    if overrides:
        new_conf = overrides
    else:
        new_conf = old_conf
    
    temp_lib_dir: Optional[Path] = None

    
    new_conf.env["BRMSPY_AUTOLOAD"] = "0"

    # fresh worker with new_conf
    session.restart(environment_conf=new_conf)

    ctx = EnvContext(session=session)

    yield ctx
