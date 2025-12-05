# functions list (based on 0.1.13)

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
brmspy.binaries._load_config
brmspy.binaries._save_config
brmspy.binaries.get_active_runtime
brmspy.binaries._set_active_runtime
brmspy.binaries._clear_active_runtime_config_only
brmspy.binaries._clear_active_runtime 

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

# Issues

