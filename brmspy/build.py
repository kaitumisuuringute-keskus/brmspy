import argparse
from pathlib import Path
from brmspy import brms
from brmspy.helpers.log import log


def main():
    """
    CLI entry point for building brmspy prebuilt runtime bundles.

    Command-line tool that orchestrates the complete runtime build process:
    1. Collect R environment metadata
    2. Stage runtime directory tree
    3. Pack into distributable archive
    """

    parser = argparse.ArgumentParser(
        description="Build brmspy prebuilt runtime bundle."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runtime_build",
        help="Directory where the runtime tree and archive will be written.",
    )
    parser.add_argument(
        "--runtime-version",
        type=str,
        default="0.1.0",
        help="Logical runtime schema/version identifier (not necessarily pip version).",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir).resolve()
    out_dir = base_dir
    runtime_version = args.runtime_version

    # install_package_deps("brms", include_suggests=False)
    with brms.manage() as ctx:
        ctx.install_rpackage("StanHeaders")

    with brms._build() as ctx:
        log("[meta] Collecting R / brms / cmdstanr metadata via rpy2...")
        metadata = ctx.collect_runtime_metadata()

        log("[stage] Staging runtime tree...")
        runtime_root = ctx.stage_runtime_tree(base_dir, metadata, runtime_version)

        log("[pack] Packing runtime to tar.gz...")
        archive_path = ctx.pack_runtime(runtime_root, out_dir, runtime_version)

        log(f"[done] Runtime bundle created: {archive_path}")


if __name__ == "__main__":
    main()
