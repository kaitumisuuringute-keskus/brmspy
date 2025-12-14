"""
Build manifest metadata collection from R environment.
"""

import json
import os
from typing import List, cast


def _run_r_json(code: str) -> dict:
    """Execute R code and parse JSON output to Python dictionary."""
    import rpy2.robjects as ro

    res = cast(List[str], ro.r(code))
    json_str = res[0]
    return json.loads(json_str)


def collect_runtime_metadata() -> dict:
    """
    Collect comprehensive R environment metadata for runtime bundle.

    Queries R via rpy2 to gather complete information about the current
    R installation, including R version, CmdStan installation, and full
    dependency closure of brms + cmdstanr with all package details.

    Returns
    -------
    dict
        Metadata dictionary containing:
        - r_version : str - R version (e.g., "4.3.1")
        - cmdstan_path : str - Path to CmdStan installation
        - cmdstan_version : str - CmdStan version
        - packages : list of dict - Package information
    """
    import rpy2.robjects as ro

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)

    # Read build-manifest.R from build directory
    manifest_r_path = os.path.join(script_dir, "build-manifest.R")
    with open(manifest_r_path, "r") as f:
        r_code = f.read()

    # Make sure jsonlite is available
    ro.r(
        'if (!requireNamespace("jsonlite", quietly = TRUE)) '
        'install.packages("jsonlite", repos="https://cloud.r-project.org")'
    )

    return _run_r_json(r_code)
