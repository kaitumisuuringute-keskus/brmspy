"""
Surface module for brms._build() context (internal tooling).

This module is intended to be imported safely in the main process (no top-level
rpy2 imports). Heavy work should be performed in the worker when these functions
are called via ModuleProxy.

Only names listed in __all__ will be exposed by the _build() context proxy.
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["BuildModule"]


class BuildModule:
    @staticmethod
    def is_package_installed(name: str) -> bool:
        from brmspy._runtime._r_packages import is_package_installed

        return is_package_installed(name)

    @staticmethod
    def collect_runtime_metadata() -> dict:
        """
        Collect comprehensive R environment metadata for runtime bundle.

        Executed in worker.
        """
        from brmspy._build._metadata import collect_runtime_metadata as _collect

        return _collect()

    @staticmethod
    def stage_runtime_tree(
        base_dir: Path, metadata: dict, runtime_version: str
    ) -> Path:
        """
        Stage runtime tree structure for packing.

        Executed in worker.
        """
        from brmspy._build._stage import stage_runtime_tree as _stage

        return _stage(base_dir, metadata, runtime_version)

    @staticmethod
    def pack_runtime(runtime_root: Path, out_dir: Path, runtime_version: str) -> Path:
        """
        Pack staged runtime into distributable archive.

        Executed in worker.
        """
        from brmspy._build._pack import pack_runtime as _pack

        return _pack(runtime_root, out_dir, runtime_version)
