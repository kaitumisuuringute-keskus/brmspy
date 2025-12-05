# functions list (based on dev after 0.1.13)

brmspy.install._init - add cran repo
brmspy.install._get_linux_repo - get Posit Package Manager (P3M) - precompiled R packages for Linux
brmspy.install._install_rpackage - installs rpackage, can have version specified
brmspy.install._install_rpackage_deps - installs all deps of an rpackage (except suggested)
brmspy.install._build_cmstanr - build cmdstanr from scratch
brmspy.install.install_prebuilt - download, extract and activate runtime.
brmspy.install.install_brms - either calls install_prebuilt or does the whole get rtools (windows), install cmdstanr, rtstan, brms etc
brmspy.install.get_brms_version

brmspy.helpers.singleton._get_brms
brmspy.helpers.singleton._get_rstan
brmspy.helpers.singleton._get_cmdstanr
brmspy.helpers.singleton._invalidate_singletons

brmspy.helpers.rtools._get_r_version
brmspy.helpers.rtools.pick_rtools_for_r
brmspy.helpers.rtools._silent_install_exe
brmspy.helpers.rtools._parse_gxx_version
brmspy.helpers.rtools._windows_has_rtools
brmspy.helpers.rtools._install_rtools_for_current_r

brmspy.binaries.config._get_config_path
brmspy.binaries.config._load_config
brmspy.binaries.config._save_config
brmspy.binaries.config.get_active_runtime
brmspy.binaries.config._set_active_runtime
brmspy.binaries.config._clear_active_runtime_config_only
brmspy.binaries.config._clear_active_runtime 

brmspy.binaries.env.system_fingerprint
brmspy.binaries.env._normalized_os_arch - normalized (os, arch)
brmspy.binaries.env.get_r_version_tuple - (major, minor, patch)
brmspy.binaries.env.r_available_and_supported
brmspy.binaries.env.extract_glibc_version
brmspy.binaries.env.parse_clang_version
brmspy.binaries.env.linux_can_use_prebuilt
brmspy.binaries.env.macos_can_use_prebuilt
brmspy.binaries.env.windows_can_use_prebuilt
brmspy.binaries.env.supported_platform -> bool
brmspy.binaries.env.toolchain_is_compatible
brmspy.binaries.env.prebuilt_available_for(fingerprint)
brmspy.binaries.env.can_use_prebuilt - UNUSED

brmspy.binaries.r._try_force_unload_package(pkg)
brmspy.binaries.r._forward_github_token_to_r
brmspy.binaries.r._get_r_pkg_version(pkg)
brmspy.binaries.r._get_r_pkg_installed(pkg)
brmspy.binaries.r._r_namespace_loaded(pkg)
brmspy.binaries.r._r_package_attached(pkg)

brmspy.binaries.use._store_default_renv
brmspy.binaries.use.activate_runtime
brmspy.binaries.use.deactivate_runtime
brmspy.binaries.use._ensure_base_dir
brmspy.binaries.use._runtime_root_for_system
brmspy.binaries.use._resolve_source
brmspy.binaries.use._is_runtime_dir
brmspy.binaries.use._load_manifest
brmspy.binaries.use._read_stored_hash
brmspy.binaries.use._write_stored_hash
brmspy.binaries.use._maybe_reuse_existing_runtime
brmspy.binaries.use._install_from_archive
brmspy.binaries.use._install_from_directory
brmspy.binaries.use.install_and_activate_runtime
brmspy.binaries.use.autoload_last_runtime

# Runtime / install subsystem design rules

1. No hidden side effects. Every function must either be:
    - Pure-ish (only computes/reads), or
    - An orchestrator whose side effects are obvious from its name and docstring.
    - “Surprise” behavior (e.g. a get_* function that also installs something) is forbidden.
    - If a step fill corrupt the current environment in case "something isnt done", it must error instead of doing "something"

2. Single responsibility per function
    - A function either does one atomic step, or coordinates a small, clearly documented flow.
    - If you can’t describe its purpose in one short sentence, split it.

3. Logical structure, not helpers spaghetti
    - Helpers that do similar things live together in one module.
    - There must be exactly one obvious place to look for:
        - runtime config persistence
        - environment detection
        - installation logic
        - runtime activation/deactivation

4. Consistent naming conventions
    - get_* / load_* / read_* → read-only.
    - save_* / write_* → disk only.
    - install_* → download/build/install on disk only.
    - activate_* / deactivate_* → mutate process/R environment only.
    - is_* / has_* / can_* → return booleans, no side effects.
    - ensure_* → may perform install/set-up, but must say so in docstring.

5. Layered, non-circular hierarchy
    - Lower layers know nothing about higher ones.
    - Each subsystem (config/env/install/use) has a single entry point; don’t call low-level pieces from random places.
    -  No circular imports, no “reach-around” calls into peers.

6. Small public API surface
    - At most 4 public functions for runtimes (installation / activation / deactivation / query).
    - Everything else is internal; it may change without notice.
    - Public functions never expose internal data structures directly.

# Current flow

## Entry Points (Public API)

From brmspy/__init__.py and brms.py:
- install_brms() - traditional R package installation
- install_prebuilt() - prebuilt runtime installation
- install_and_activate_runtime() - low-level runtime installer
- activate_runtime() - activate existing runtime
- deactivate_runtime() - deactivate runtime
- fit() / brm() - fit Bayesian models
- On import: autoload_last_runtime() runs automatically

## Flow 1: Traditional Install (install_brms with use_prebuilt_binaries=False)

install.install_brms()
├─> install._init() - sets CRAN mirror
├─> [if install_rtools] rtools._install_rtools_for_current_r()
│   ├─> rtools._get_r_version()
│   ├─> rtools.pick_rtools_for_r() - map R version to Rtools
│   ├─> rtools._windows_has_rtools() - check if present
│   └─> [if not present] rtools._silent_install_exe() - downloads and installs
├─> r._forward_github_token_to_r()
├─> install._install_rpackage("brms", version)
│   ├─> r._get_r_pkg_installed() - check if already installed
│   ├─> [if version specified] uses remotes::install_version()
│   ├─> [else] uses utils.install_packages()
│   └─> [on Linux] prepends P3M repo from install._get_linux_repo()
├─> install._install_rpackage_deps("brms")
├─> [if install_cmdstanr] install._install_rpackage("cmdstanr", version)
├─> [if install_cmdstanr] install._install_rpackage_deps("cmdstanr")
├─> [if install_cmdstanr] install._build_cmdstanr()
│   ├─> [on Windows] cmdstanr::check_cmdstan_toolchain()
│   └─> cmdstanr::install_cmdstan()
├─> [if install_rstan] install._install_rpackage("rstan", version)
├─> [if install_rstan] install._install_rpackage_deps("rstan")
├─> singleton._invalidate_singletons()
└─> singleton._get_brms() - imports and caches R packages
    ├─> importr("cmdstanr") [optional, may be None]
    ├─> importr("rstan") [optional, may be None]
    ├─> importr("posterior") [required]
    ├─> importr("brms") [required]
    └─> importr("base") [required]

## Flow 2: Prebuilt Runtime Install

install.install_prebuilt()
├─> install._init()
├─> r._forward_github_token_to_r()
├─> [if install_rtools] rtools._install_rtools_for_current_r()
├─> env.can_use_prebuilt() - verify system compatibility
│   ├─> env.supported_platform() - check OS/arch (linux/macos/windows, x86_64/arm64)
│   ├─> env.r_available_and_supported() - check R >= 4.0
│   └─> env.toolchain_is_compatible()
│       ├─> [linux] env.linux_can_use_prebuilt() - check glibc >= 2.27, g++ >= 9
│       ├─> [macos] env.macos_can_use_prebuilt() - check Xcode CLT, clang >= 11
│       └─> [windows] env.windows_can_use_prebuilt() - check Rtools/MinGW
├─> env.system_fingerprint() - get {os}-{arch}-r{major}.{minor}
├─> construct GitHub release URL
└─> use.install_and_activate_runtime() [see Flow 3]

## Flow 3: Runtime Installation & Activation

use.install_and_activate_runtime()
├─> [if official GitHub URL] github.get_github_asset_sha256_from_url()
│   ├─> github._parse_github_release_download_url()
│   ├─> github._github_get_json() - fetch from GitHub API
│   └─> extracts digest from asset metadata
├─> use._ensure_base_dir() - creates ~/.brmspy/runtime
├─> use._runtime_root_for_system() - determines install path
├─> use._maybe_reuse_existing_runtime() - check if already installed
│   ├─> use._is_runtime_dir() - validate structure
│   ├─> use._read_stored_hash() - read hash file
│   ├─> compare hashes
│   └─> [if match and activate] use.activate_runtime() [see Flow 4]
├─> [if not reusing] use._resolve_source() - download or locate bundle
│   └─> [if URL] urllib.request.urlretrieve() to temp file
├─> [if directory] use._install_from_directory()
│   ├─> use._load_manifest() - validate manifest.json
│   └─> shutil.move() to runtime_root
├─> [if archive] use._install_from_archive()
│   ├─> tarfile.extractall() to temp dir
│   ├─> use._load_manifest()
│   └─> shutil.move() to runtime_root
├─> use._write_stored_hash() - save hash for future reuse
├─> use._load_manifest() - final validation
└─> [if activate] use.activate_runtime() [see Flow 4]

## Flow 4: Runtime Activation

use.activate_runtime(runtime_root)
├─> validate structure: manifest.json, Rlib/, cmdstan/ exist
├─> use._load_manifest()
├─> [optional] verify fingerprint matches env.system_fingerprint()
├─> use._store_default_renv() - save original .libPaths() and cmdstan_path
├─> singleton._invalidate_singletons()
├─> for pkg in ["brms", "cmdstanr", "rstan"]:
│   └─> [if loaded] r._try_force_unload_package(pkg) - aggressive unload
├─> ro.r('.libPaths(c("{rlib_posix}"))') - set R library path
├─> ro.r('cmdstanr::set_cmdstan_path("{cmdstan_posix}")') - set CmdStan path
├─> verify brms and cmdstanr loadable
├─> config._set_active_runtime(runtime_root) - save to ~/.brmspy/config.json
└─> log.greet()

## Flow 5: Auto-Activation on Import

import brmspy / from brmspy import brms
├─> brmspy/__init__.py imports brmspy.brms
└─> brms.py module level:
    ├─> use.autoload_last_runtime()
    │   ├─> config.get_active_runtime() - read ~/.brmspy/config.json
    │   ├─> [if exists] use.activate_runtime() [see Flow 4]
    │   └─> [if error] config._set_active_runtime(None) - clear invalid config
    └─> singleton._get_brms() - import R packages [see Flow 1]

## Flow 6: Deactivation

use.deactivate_runtime()
├─> check if _default_libpath exists and runtime active
├─> for pkg in ["brms", "cmdstanr", "rstan"]:
│   └─> [if loaded] r._try_force_unload_package(pkg)
├─> ro.r('.libPaths()') - restore to _default_libpath
├─> [if _default_cmdstan_path] restore cmdstanr path
├─> [else] ro.r('cmdstanr::set_cmdstan_path(path=NULL)')
├─> config._clear_active_runtime_config_only()
└─> singleton._invalidate_singletons()

## Flow 7: Building Runtime (for developers)

build.main() or collect_runtime_metadata + stage_runtime_tree + pack_runtime
├─> build.collect_runtime_metadata() - executes build-manifest.R
│   └─> returns {r_version, cmdstan_path, cmdstan_version, packages: [{Package, Version, LibPath}]}
├─> build.stage_runtime_tree(base_dir, metadata, runtime_version)
│   ├─> env.system_fingerprint() - determine target platform
│   ├─> create {base_dir}/{fingerprint}/ structure
│   ├─> copy all R packages to Rlib/
│   ├─> copy CmdStan to cmdstan/
│   ├─> build._generate_manifest_hash() - SHA256 of manifest
│   └─> write manifest.json with metadata + hash
└─> build.pack_runtime(runtime_root, out_dir, runtime_version)
    └─> create brmspy-runtime-{version}-{fingerprint}.tar.gz

## Helper Module Functions

### Config Management (binaries/config.py)
- _get_config_path() → ~/.brmspy/config.json
- _load_config() → dict
- _save_config(dict) → atomic write
- get_active_runtime() → Optional[Path]
- _set_active_runtime(Path) → saves to config
- _clear_active_runtime() → calls deactivate_runtime() + clears config

### Environment Detection (binaries/env.py)
- system_fingerprint() → "{os}-{arch}-r{major}.{minor}"
- _normalized_os_arch() → ("linux"|"macos"|"windows", "x86_64"|"arm64")
- get_r_version_tuple() → (major, minor, patch)
- r_available_and_supported(min_major=4, min_minor=0) → bool
- extract_glibc_version(ldd_output) → (major, minor)
- parse_clang_version(clang_output) → (major, minor)
- linux_can_use_prebuilt() → checks glibc >= 2.27, g++ >= 9
- macos_can_use_prebuilt() → checks Xcode CLT, clang >= 11
- windows_can_use_prebuilt() → checks Rtools/MinGW g++ >= 9
- supported_platform() → checks OS/arch in whitelist
- toolchain_is_compatible() → routes to OS-specific check
- can_use_prebuilt() → master check (platform + R + toolchain)
- prebuilt_available_for(fingerprint) → checks PREBUILT_FINGERPRINTS set

### R Package Management (binaries/r.py)
- _try_force_unload_package(pkg) → detach, unloadNamespace, pkgload::unload, library.dynam.unload
- _forward_github_token_to_r() → Sys.setenv(GITHUB_PAT/TOKEN)
- _get_r_pkg_version(pkg) → Optional[Version]
- _get_r_pkg_installed(pkg) → bool
- _r_namespace_loaded(pkg) → bool
- _r_package_attached(pkg) → bool

### GitHub Integration (binaries/github.py)
- _parse_github_release_download_url(url) → (owner, repo, tag, asset)
- _github_get_json(api_url) → dict, handles auth + retries
- get_github_release_asset_metadata_from_url(url) → GitHubReleaseAssetMetadata
- get_github_asset_sha256_from_url(url) → Optional[str]

### Rtools Management (helpers/rtools.py)
- _get_r_version() → Version
- pick_rtools_for_r(r_ver) → "40"|"42"|"43"|"44"|"45"|None
- _silent_install_exe(url, label) → downloads and installs .exe
- _parse_gxx_version(g++_output) → (major, minor)
- _windows_has_rtools() → checks for make + mingw g++
- _install_rtools_for_current_r() → auto-detect + install + update PATH

### Singleton Management (helpers/singleton.py)
- _get_brms() → cached brms package, imports on first call
- _get_base() → cached base R package
- _get_rstan() → cached rstan or None
- _get_cmdstanr() → cached cmdstanr or None
- _invalidate_singletons() → clears all caches

## Key Data Structures

### Runtime Manifest (manifest.json)
```json
{
  "runtime_version": "0.1.0",
  "fingerprint": "linux-x86_64-r4.3",
  "r_version": "4.3.2",
  "cmdstan_version": "2.33.1",
  "r_packages": {"brms": "2.21.0", "cmdstanr": "0.8.1", ...},
  "manifest_hash": "sha256...",
  "built_at": "2024-01-01T00:00:00Z"
}
```

### Config File (~/.brmspy/config.json)
```json
{
  "active_runtime": "/home/user/.brmspy/runtime/linux-x86_64-r4.3-0.1.0"
}
```

### Runtime Directory Structure
```
~/.brmspy/runtime/{fingerprint}-{version}/
├── manifest.json
├── hash (SHA256 of bundle)
├── Rlib/
│   ├── brms/
│   ├── cmdstanr/
│   ├── posterior/
│   └── ... (all dependencies)
└── cmdstan/
    ├── bin/
    ├── stan/
    └── ...
```

# Issues

