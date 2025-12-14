"""
Tests for brms family functions.

These tests verify:
- brmsfamily() core function works
- Family wrapper functions construct properly
- Families can be used with fit()
- Family object inspection
"""

import pytest


@pytest.mark.requires_brms
class TestBrmsfamilyCore:
    """Test the core brmsfamily() function."""

    def test_brmsfamily_basic(self):
        """Test basic brmsfamily() usage"""

        from brmspy.types.session import SexpWrapper
        from brmspy.brms import brmsfamily

        # Create basic Gaussian family
        family = brmsfamily("gaussian", link="identity")

        # Verify return type
        assert isinstance(family, SexpWrapper)
        assert family is not None

    def test_brmsfamily_with_links(self):
        """Test brmsfamily() with various link functions"""
        from brmspy.brms import brmsfamily
        from brmspy.types.session import SexpWrapper

        # Test with different links
        family_log = brmsfamily("poisson", link="log")
        family_logit = brmsfamily("binomial", link="logit")
        family_probit = brmsfamily("binomial", link="probit")

        assert isinstance(family_log, SexpWrapper)
        assert isinstance(family_logit, SexpWrapper)
        assert isinstance(family_probit, SexpWrapper)

    def test_brmsfamily_with_auxiliary_links(self):
        """Test brmsfamily() with auxiliary parameter links"""
        from brmspy.brms import brmsfamily
        from brmspy.types.session import SexpWrapper

        # Gaussian with sigma link
        family = brmsfamily("gaussian", link="identity", link_sigma="log")

        assert isinstance(family, SexpWrapper)

        # Negative binomial with shape link
        family_nb = brmsfamily("negbinomial", link="log", link_shape="log")

        assert isinstance(family_nb, SexpWrapper)


@pytest.mark.requires_brms
class TestFamilyWrappers:
    """Test that all family wrapper functions construct properly."""

    def test_all_families_construct(self):
        """Test that all family wrapper functions can be instantiated"""
        from brmspy import brms
        from brmspy.types.session import SexpWrapper

        # List of all family functions to test
        family_functions = [
            ("gaussian", {}),
            ("student", {}),
            ("poisson", {}),
            ("binomial", {}),
            ("bernoulli", {}),
            ("beta_binomial", {}),
            ("negbinomial", {}),
            ("geometric", {}),
            ("lognormal", {}),
            ("shifted_lognormal", {}),
            ("skew_normal", {}),
            ("exponential", {}),
            ("weibull", {}),
            ("frechet", {}),
            ("gen_extreme_value", {}),
            ("exgaussian", {}),
            ("wiener", {}),
            ("Beta", {}),
            ("xbeta", {}),
            ("Gamma", {}),
            ("inverse_gaussian", {}),
            ("dirichlet", {"refcat": None}),
            ("logistic_normal", {"refcat": None}),
            ("von_mises", {}),
            ("asym_laplace", {}),
            ("cox", {}),
            ("hurdle_poisson", {}),
            ("hurdle_negbinomial", {}),
            ("hurdle_gamma", {}),
            ("hurdle_lognormal", {}),
            ("hurdle_cumulative", {}),
            ("zero_inflated_beta", {}),
            ("zero_one_inflated_beta", {}),
            ("zero_inflated_poisson", {}),
            ("zero_inflated_negbinomial", {}),
            ("zero_inflated_binomial", {}),
            ("zero_inflated_beta_binomial", {}),
            ("categorical", {"refcat": None}),
            ("multinomial", {"refcat": None}),
            ("dirichlet_multinomial", {"refcat": None}),
            ("cumulative", {}),
            ("sratio", {}),
            ("cratio", {}),
            ("acat", {}),
        ]

        # Test each family function
        for family_name, kwargs in family_functions:
            family_fn = getattr(brms, family_name)
            family_obj = family_fn(**kwargs)

            assert isinstance(
                family_obj, SexpWrapper
            ), f"{family_name}() should return SexpWrapper"
            assert family_obj is not None, f"{family_name}() should not return None"

    def test_common_families_with_custom_links(self):
        """Test common families with custom link functions"""
        from brmspy.brms import gaussian, poisson, binomial
        from brmspy.types.session import SexpWrapper

        # Gaussian with log link (non-standard)
        gauss_log = gaussian(link="log")
        assert isinstance(gauss_log, SexpWrapper)

        # Poisson with identity link (non-standard)
        pois_identity = poisson(link="identity")
        assert isinstance(pois_identity, SexpWrapper)

        # Binomial with probit link
        binom_probit = binomial(link="probit")
        assert isinstance(binom_probit, SexpWrapper)


@pytest.mark.requires_brms
@pytest.mark.slow
class TestFamilyIntegration:
    """Test families integrated with model fitting."""

    def test_gaussian_family_with_fit(self, sample_dataframe):
        """Test using gaussian() family with fit()"""
        from brmspy import brms
        from brmspy.brms import gaussian
        import arviz as az

        # Create family object
        family_obj = gaussian()

        # Use with fit - note brms.fit() expects family as string
        # but we can verify the family object is valid
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family=family_obj,
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        assert isinstance(model.idata, az.InferenceData)

    def test_poisson_family_with_fit(self, poisson_data):
        """Test using poisson() family with fit()"""
        from brmspy import brms
        from brmspy.brms import poisson
        import arviz as az

        # Create family object
        family_obj = poisson()

        # Fit model with Poisson family
        model = brms.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family=family_obj,
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        assert isinstance(model.idata, az.InferenceData)

        # Verify it's a count model
        param_names = list(model.idata.posterior.data_vars)
        assert len(param_names) > 0

    def test_student_family_robust_regression(self, sample_dataframe):
        """Test student() family for robust regression"""
        from brmspy import brms
        from brmspy.brms import student
        import arviz as az
        from brmspy.types.session import SexpWrapper

        # Create student family object
        family_obj = student()
        assert isinstance(family_obj, SexpWrapper)

        # Fit robust regression
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family=family_obj,
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        assert isinstance(model.idata, az.InferenceData)

        # Student-t should have nu (degrees of freedom) parameter
        param_names = list(model.idata.posterior.data_vars)
        assert any(
            "nu" in p.lower() for p in param_names
        ), "Student-t model should have nu parameter"


@pytest.mark.requires_brms
class TestFamilyExtraction:
    """Test extracting family from fitted models."""

    @pytest.mark.slow
    def test_family_extraction(self, sample_dataframe):
        """Test family() function to extract family from fitted model"""
        from brmspy import brms
        from brmspy.types.brms_results import ProxyListSexpVector

        # Fit a model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Extract family from fitted model
        family_obj = brms.family(model)

        # Verify return type
        assert isinstance(family_obj, ProxyListSexpVector)
        assert family_obj is not None


@pytest.mark.requires_brms
class TestSpecialFamilies:
    """Test special family types (zero-inflated, hurdle, ordinal)."""

    def test_zero_inflated_families(self):
        """Test zero-inflated family constructors"""
        from brmspy.brms import (
            zero_inflated_poisson,
            zero_inflated_negbinomial,
            zero_inflated_binomial,
            zero_inflated_beta,
        )
        from brmspy.types.session import SexpWrapper

        # Test each zero-inflated family
        zi_pois = zero_inflated_poisson()
        zi_nb = zero_inflated_negbinomial()
        zi_binom = zero_inflated_binomial()
        zi_beta = zero_inflated_beta()

        assert isinstance(zi_pois, SexpWrapper)
        assert isinstance(zi_nb, SexpWrapper)
        assert isinstance(zi_binom, SexpWrapper)
        assert isinstance(zi_beta, SexpWrapper)

    def test_hurdle_families(self):
        """Test hurdle family constructors"""
        from brmspy.brms import (
            hurdle_poisson,
            hurdle_negbinomial,
            hurdle_gamma,
            hurdle_lognormal,
        )
        from brmspy.types.session import SexpWrapper

        # Test each hurdle family
        h_pois = hurdle_poisson()
        h_nb = hurdle_negbinomial()
        h_gamma = hurdle_gamma()
        h_lognorm = hurdle_lognormal()

        assert isinstance(h_pois, SexpWrapper)
        assert isinstance(h_nb, SexpWrapper)
        assert isinstance(h_gamma, SexpWrapper)
        assert isinstance(h_lognorm, SexpWrapper)

    def test_ordinal_families(self):
        """Test ordinal model family constructors"""
        from brmspy.brms import cumulative, sratio, cratio, acat
        from brmspy.types.session import SexpWrapper

        # Test each ordinal family
        ord_cum = cumulative()
        ord_sr = sratio()
        ord_cr = cratio()
        ord_ac = acat()

        assert isinstance(ord_cum, SexpWrapper)
        assert isinstance(ord_sr, SexpWrapper)
        assert isinstance(ord_cr, SexpWrapper)
        assert isinstance(ord_ac, SexpWrapper)

    def test_categorical_families(self):
        """Test categorical/multinomial family constructors"""
        from brmspy.brms import (
            categorical,
            multinomial,
            dirichlet_multinomial,
        )
        from brmspy.types.session import SexpWrapper

        # Test categorical families
        cat = categorical()
        multi = multinomial()
        dir_multi = dirichlet_multinomial()

        assert isinstance(cat, SexpWrapper)
        assert isinstance(multi, SexpWrapper)
        assert isinstance(dir_multi, SexpWrapper)


@pytest.mark.requires_brms
class TestFamilyParameters:
    """Test family-specific parameters and link functions."""

    def test_beta_family_parameters(self):
        """Test Beta family with precision parameter"""
        from brmspy.brms import Beta
        from brmspy.types.session import SexpWrapper

        # Beta with default parameters
        beta_default = Beta()
        assert isinstance(beta_default, SexpWrapper)

        # Beta with custom precision link
        beta_custom = Beta(link="logit", link_phi="identity")
        assert isinstance(beta_custom, SexpWrapper)

    def test_gamma_family_parameters(self):
        """Test Gamma family with shape parameter"""
        from brmspy.brms import Gamma
        from brmspy.types.session import SexpWrapper

        # Gamma with default parameters
        gamma_default = Gamma()
        assert isinstance(gamma_default, SexpWrapper)

        # Gamma with custom shape link
        gamma_custom = Gamma(link="log", link_shape="identity")
        assert isinstance(gamma_custom, SexpWrapper)

    def test_weibull_family_parameters(self):
        """Test Weibull family for survival analysis"""
        from brmspy.brms import weibull
        from brmspy.types.session import SexpWrapper

        # Weibull with default parameters
        weib = weibull()
        assert isinstance(weib, SexpWrapper)

        # Weibull with custom shape link
        weib_custom = weibull(link="log", link_shape="identity")
        assert isinstance(weib_custom, SexpWrapper)

    def test_threshold_parameter_ordinal(self):
        """Test threshold parameter for ordinal families"""
        from brmspy.brms import cumulative
        from brmspy.types.session import SexpWrapper

        # Test different threshold types
        cum_flex = cumulative(threshold="flexible")
        cum_equi = cumulative(threshold="equidistant")

        assert isinstance(cum_flex, SexpWrapper)
        assert isinstance(cum_equi, SexpWrapper)

    def test_refcat_parameter(self):
        """Test reference category parameter"""
        from brmspy.brms import categorical, multinomial
        from brmspy.types.session import SexpWrapper

        # Test with default refcat (None)
        cat_default = categorical()
        assert isinstance(cat_default, SexpWrapper)

        # Test with explicit refcat
        cat_ref = categorical(refcat="category1")
        assert isinstance(cat_ref, SexpWrapper)

        multi_ref = multinomial(refcat="ref_level")
        assert isinstance(multi_ref, SexpWrapper)


@pytest.mark.requires_brms
class TestExoticFamilies:
    """Test less common but important families."""

    def test_wiener_diffusion_model(self):
        """Test Wiener diffusion model for reaction time"""
        from brmspy.brms import wiener
        from brmspy.types.session import SexpWrapper

        # Wiener with all parameters
        wien = wiener(link="identity", link_bs="log", link_ndt="log", link_bias="logit")

        assert isinstance(wien, SexpWrapper)

    def test_von_mises_circular(self):
        """Test von Mises for circular/directional data"""
        from brmspy.brms import von_mises
        from brmspy.types.session import SexpWrapper

        # Von Mises with default parameters
        vm = von_mises()
        assert isinstance(vm, SexpWrapper)

        # Von Mises with custom concentration link
        vm_custom = von_mises(link="tan_half", link_kappa="identity")
        assert isinstance(vm_custom, SexpWrapper)

    def test_cox_survival_model(self):
        """Test Cox proportional hazards model"""
        from brmspy.brms import cox
        from brmspy.types.session import SexpWrapper

        # Cox model
        cox_model = cox()
        assert isinstance(cox_model, SexpWrapper)

    def test_exgaussian_reaction_time(self):
        """Test ex-Gaussian for reaction time data"""
        from brmspy.brms import exgaussian
        from brmspy.types.session import SexpWrapper

        # Ex-Gaussian with all parameters
        exg = exgaussian(link="identity", link_sigma="log", link_beta="log")

        assert isinstance(exg, SexpWrapper)

    def test_asymmetric_laplace_quantile(self):
        """Test asymmetric Laplace for quantile regression"""
        from brmspy.brms import asym_laplace
        from brmspy.types.session import SexpWrapper

        # Asymmetric Laplace with quantile parameter
        asym_lap = asym_laplace(
            link="identity", link_sigma="log", link_quantile="logit"
        )

        assert isinstance(asym_lap, SexpWrapper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
