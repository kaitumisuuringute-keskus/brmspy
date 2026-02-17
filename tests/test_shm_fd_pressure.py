"""
Test for SHM file descriptor leak.

The core fix: ShmPool now uses a slab allocator (open_slab/seal_slab)
to pack hundreds of sub-allocations into a single SharedMemory block,
reducing the fd count from O(n_arrays) to O(1) per encode call.
"""

import gc
import os
import platform

import pytest


def _count_open_fds() -> int:
    """Return the number of open file descriptors for this process."""
    if platform.system() == "Darwin":
        return len(os.listdir("/dev/fd"))
    elif platform.system() == "Linux":
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    else:
        pytest.skip("FD counting not supported on this platform")


@pytest.mark.requires_brms
class TestShmFdPressure:

    @pytest.mark.slow
    def test_fd_leak_under_repeated_fit_summary_predict(self, sample_dataframe):
        """
        Stress test: fit + summary + posterior_linpred in a loop with a
        very wide dataframe (~2500 columns).  Without the slab allocator,
        this would create 2500+ SHM blocks per encode call and blow the
        macOS fd limit (256) instantly.
        """
        import numpy as np
        from brmspy import brms

        n_iterations = 4
        fd_counts = []

        # Build an extremely wide dataframe to stress SHM allocation
        # Real MaxDiff can have 500+ columns; we go even wider to be safe
        n_extra_cols = 2500
        wide_df = sample_dataframe.copy()
        for i in range(n_extra_cols):
            wide_df[f"noise_{i}"] = np.random.randn(len(wide_df))

        # Warm-up: one fit to stabilize baseline FDs
        warmup_model = brms.fit(
            formula="y ~ x1",
            data=wide_df,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=2,
            silent=2,
            refresh=0,
        )
        _ = brms.summary(warmup_model)
        del warmup_model, _
        gc.collect()
        gc.collect()

        baseline_fds = _count_open_fds()
        print(f"\nBaseline FDs after warmup: {baseline_fds}")

        for i in range(n_iterations):
            model = brms.fit(
                formula="y ~ x1",
                data=wide_df,
                family="gaussian",
                iter=100,
                warmup=50,
                chains=2,
                silent=2,
                refresh=0,
            )
            summary = brms.summary(model)
            pred = brms.posterior_linpred(model, newdata=wide_df)

            del model, summary, pred
            gc.collect()
            gc.collect()

            current_fds = _count_open_fds()
            fd_counts.append(current_fds)
            print(f"  Iteration {i+1}: {current_fds} FDs (delta from baseline: +{current_fds - baseline_fds})")

        total_growth = fd_counts[-1] - baseline_fds
        print(f"\n  Total FD growth from baseline: {total_growth}")

        assert total_growth < 30, (
            f"FD leak detected: {total_growth} new FDs accumulated over "
            f"{n_iterations} iterations (baseline={baseline_fds}, "
            f"final={fd_counts[-1]}, per-iter={fd_counts}). "
            f"SHM handles are likely not being closed on object GC."
        )
