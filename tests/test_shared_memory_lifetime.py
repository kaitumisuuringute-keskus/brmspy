import pytest


@pytest.mark.requires_brms
class TestSharedMemoryLifetime:

    @pytest.mark.slow
    def test_shared_memory_usable_after_restart(self, sample_dataframe):
        """Test shm stays usable after restarting R"""
        from brmspy import brms
        import gc

        models = []
        for _ in range(2):
            with brms.manage(environment_name="_test"):
                # Triggers a restart and SharedMemoryManager shutdown
                pass
            model = brms.fit(
                formula="y ~ x1",
                data=sample_dataframe,
                family="gaussian",
                iter=100,
                warmup=50,
                chains=2,
                silent=2,
                refresh=0,
            )
            models.append(model)

        # Try to be as mean as possible to GC
        gc.collect()

        # Accessing idata must not crash; if SHM was closed, this will instantly die
        id0 = models[0].idata
        id1 = models[1].idata

        # Touch the actual SHM-backed arrays
        _ = id0.posterior.dims
        _ = id1.posterior.dims

        # Force a repr, which walks the Dataset tree and touches data buffers
        assert "posterior" in repr(id0)
        assert "posterior" in repr(id1)
