"""
Prebuilt Runtime Binary Management
===================================

This subpackage provides functionality for working with prebuilt brms runtime
bundles - self-contained environments with pre-installed brms, cmdstanr, and
compiled Stan models for fast installation and consistent execution.

Main Functions
--------------
install_and_activate_runtime
    Download, extract, and activate a prebuilt runtime bundle
activate_runtime
    Activate an already-installed runtime bundle
system_fingerprint
    Get system identifier for runtime bundle selection
get_active_runtime
    Get path to currently saved active runtime
set_active_runtime
    Save a runtime path as active for auto-activation
clear_active_runtime
    Clear saved runtime configuration

Modules
-------
use
    Runtime installation and activation
env
    Platform detection and compatibility checking
build
    Runtime bundle creation (for developers/CI)

Prebuilt Runtimes
-----------------
Prebuilt runtime bundles provide:

- **Speed**: Install in seconds vs minutes for source builds
- **Consistency**: Identical R package versions across systems
- **Reliability**: Pre-tested, pre-compiled Stan models
- **Portability**: Works across systems with compatible toolchains

Bundle Structure
----------------
Each runtime bundle contains::

    runtime/
    ├── manifest.json         # Metadata and fingerprint
    ├── Rlib/                 # R package library
    │   ├── brms/            # brms package
    │   ├── cmdstanr/        # cmdstanr package
    │   ├── StanHeaders/     # Stan C++ headers
    │   └── ...              # Dependencies
    └── cmdstan/             # CmdStan installation
        ├── bin/             # Compiled executables
        ├── stan/            # Stan language files
        └── ...

System Fingerprints
-------------------
Runtime bundles are platform-specific, identified by fingerprints::

    {os}-{arch}-r{major}.{minor}

Examples:
- linux-x86_64-r4.3
- macos-arm64-r4.4
- windows-x86_64-r4.2

Examples
--------
Install and activate runtime from URL:

```python
from brmspy.binaries import install_and_activate_runtime

# Download and install from GitHub releases
url = "https://github.com/.../runtime-linux-x86_64-r4.3.tar.gz"
runtime_path = install_and_activate_runtime(url=url)

# Now use brms normally
from brmspy import fit
result = fit("y ~ x", data={"y": [1,2,3], "x": [1,2,3]})
```

Activate existing runtime:

```python
from pathlib import Path
from brmspy.binaries import activate_runtime

# Activate previously installed runtime
runtime_dir = Path.home() / ".brmspy" / "runtime" / "linux-x86_64-r4.3"
activate_runtime(runtime_dir)
```

Check system compatibility:

```
from brmspy.binaries import system_fingerprint
from brmspy.binaries.env import can_use_prebuilt

# Get system identifier
fp = system_fingerprint()
print(f"System: {fp}")  # e.g., "linux-x86_64-r4.3"

# Check if prebuilt bundles can be used
if can_use_prebuilt():
    print("System compatible with prebuilt bundles")
```

See Also
--------
brmspy.install.install_prebuilt : High-level installation with auto-selection
brmspy.binaries.env : Platform detection and compatibility
brmspy.binaries.use : Runtime installation functions
brmspy.binaries.build : Build runtime bundles (for developers)

Notes
-----
**For End Users:**

Most users should use the high-level `brmspy.install_prebuilt()` function
which automatically selects and installs the correct runtime for their system.

**For Developers:**

The `build` module provides tools for creating runtime bundles:
- `collect_runtime_metadata()`: Gather package info
- `stage_runtime_tree()`: Prepare bundle directory structure
- `pack_runtime()`: Create distributable archive

**Compatibility:**

Prebuilt runtimes require:
- Compatible OS and architecture
- Matching R major.minor version
- Compatible system toolchain (compilers, libraries)

See `brmspy.binaries.env` for detailed compatibility checking.
"""
from brmspy.binaries.use import (
    install_and_activate_runtime,
    activate_runtime
)
from brmspy.binaries.env import system_fingerprint
from brmspy.binaries.config import (
    get_active_runtime,
    set_active_runtime,
    clear_active_runtime
)

__all__ = [
    'install_and_activate_runtime',
    'activate_runtime',
    'system_fingerprint',
    'get_active_runtime',
    'set_active_runtime',
    'clear_active_runtime'
]