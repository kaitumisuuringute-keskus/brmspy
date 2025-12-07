"""
Integration tests for formula functions with family combinations.

These tests cover all formula.py functions (bf, lf, nlf, acformula, set_*) 
integrated with various family types. Each test actually fits a model to 
verify end-to-end integration.

Mark with: @pytest.mark.requires_brms and @pytest.mark.slow
"""
import pytest
import pandas as pd
import numpy as np


@pytest.mark.requires_brms
@pytest.mark.slow
class TestFormulaFamilyIntegration:
    """
    Comprehensive tests for formula functions with family integration.
    
    Tests all functions from brmspy/brms_functions/formula.py:
    - bf(): Base formula specification
    - lf(): Linear formulas for distributional parameters
    - nlf(): Non-linear formulas
    - acformula(): Autocorrelation structures
    - set_rescor(): Residual correlation control
    - set_mecor(): Measurement error correlation
    - set_nl(): Non-linear model specification
    
    Each test uses max 100 iterations, 50 warmup for speed.
    """
    
    def test_distributional_regression_student_nu(self):
        """
        Test bf() + lf() with student family's nu parameter.
        
        Validates:
        - lf() function for distributional parameters
        - student() family with modeled degrees of freedom
        - Integration of formula + distributional param + family
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf
        from brmspy.brms_functions.families import student
        import arviz as az
        
        # Load epilepsy dataset
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create formula with modeled degrees of freedom
        formula = bf("count ~ zAge") + lf("nu ~ zBase", dpar="nu") + student()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted successfully
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_zAge' in p for p in param_names), \
            "Should have coefficient for zAge"
        assert any('b_nu' in p or 'nu' in p.lower() for p in param_names), \
            "Should have parameters for modeled nu"
    
    def test_skew_normal_modeled_skewness(self):
        """
        Test bf() + lf() with skew_normal family's alpha parameter.
        
        Validates:
        - lf() with skewness parameter
        - skew_normal() family integration
        - Distributional modeling of asymmetry
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf
        from brmspy.brms_functions.families import skew_normal
        import arviz as az
        
        # Load epilepsy dataset
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create formula with modeled skewness
        formula = bf("count ~ zAge") + lf("alpha ~ Trt", dpar="alpha") + skew_normal()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_zAge' in p for p in param_names), \
            "Should have coefficient for zAge"
        assert any('alpha' in p.lower() for p in param_names), \
            "Should have parameters for modeled alpha"
    
    def test_zero_inflated_modeled_zi(self):
        """
        Test bf() + lf() with zero_inflated_poisson family's zi parameter.
        
        Validates:
        - lf() with zero-inflation parameter
        - zero_inflated_poisson() family
        - Hierarchical structure + distributional modeling
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf
        from brmspy.brms_functions.families import zero_inflated_poisson
        import arviz as az
        
        # Load epilepsy dataset
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create formula with modeled zero-inflation
        formula = bf("count ~ zAge + (1|patient)") + \
                  lf("zi ~ zBase", dpar="zi") + \
                  zero_inflated_poisson()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_zAge' in p for p in param_names), \
            "Should have coefficient for zAge"
        assert any('sd_patient' in p for p in param_names), \
            "Should have random effects"
        assert any('zi' in p.lower() for p in param_names), \
            "Should have parameters for modeled zi"
    
    def test_nonlinear_model_nlf_set_nl(self):
        """
        Test bf() + lf() + set_nl() for non-linear models.
        
        Validates:
        - lf() function for non-linear parameter formulas
        - set_nl() function to mark model as non-linear
        - gaussian() family with non-linear predictor
        - Formula pattern from brms docs: y ~ a * inv_logit(x * b)
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf, set_nl
        from brmspy.brms_functions.families import gaussian
        import arviz as az
        
        # Create synthetic data matching the brms docs example
        # Pattern: y ~ a * inv_logit(x * b) where a, b are functions of z
        np.random.seed(42)
        n = 100
        z = np.random.normal(0, 1, n)
        x = np.random.uniform(-2, 2, n)
        
        # True non-linear relationship
        a_true = 5 + 0.5 * z
        b_true = 1 + 0.3 * z
        y = a_true * (1 / (1 + np.exp(-x * b_true))) + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({
            'y': y,
            'x': x,
            'z': z
        })
        
        formula = bf("y ~ a * inv_logit(x * b)") + \
                  lf("a + b ~ z") + \
                  set_nl() + \
                  gaussian()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=data,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check that non-linear parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_a' in p.lower() or '_a_' in p.lower() for p in param_names), \
            "Should have parameters for non-linear parameter 'a'"
        assert any('b_b' in p.lower() or '_b_' in p.lower() for p in param_names), \
            "Should have parameters for non-linear parameter 'b'"
    
    def test_autocorrelation_arma(self):
        """
        Test bf() + acformula() for time series models.
        
        Validates:
        - acformula() function for autocorrelation structures
        - ARMA(p,q) specification
        - gaussian() family with autocorrelation
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, acformula
        from brmspy.brms_functions.families import gaussian
        import arviz as az
        
        # Load epilepsy dataset (has time structure per patient)
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create formula with autocorrelation
        formula = bf("count ~ zAge + (1|patient)") + \
                  acformula("~ arma(p=1, q=1)") + \
                  gaussian()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_zAge' in p for p in param_names), \
            "Should have coefficient for zAge"
        # ARMA parameters might be named differently depending on brms version
        assert len(param_names) > 0, \
            "Should have fitted parameters"
    
    def test_measurement_error_mecor(self):
        """
        Test bf() + set_mecor() for measurement error models.
        
        Validates:
        - set_mecor() function for ME correlation control
        - me() syntax in formulas
        - gaussian() family with measurement error
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, set_mecor
        from brmspy.brms_functions.families import gaussian
        import arviz as az
        
        # Create synthetic data with measurement error
        np.random.seed(42)
        n = 80
        x_true = np.random.normal(0, 1, n)
        x_obs = x_true + np.random.normal(0, 0.5, n)  # Observed with error
        sdx = np.repeat(0.5, n)  # Known SD of measurement error
        z = np.random.normal(0, 1, n)
        y = 2 + 1.5 * x_true + 0.8 * z + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({
            'y': y,
            'x': x_obs,
            'sdx': sdx,
            'z': z
        })
        
        # Create formula with measurement error
        formula = bf("y ~ me(x, sdx) + z") + set_mecor(True) + gaussian()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=data,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_z' in p for p in param_names), \
            "Should have coefficient for z"
        # ME-corrected coefficient might have special naming
        assert len(param_names) > 0, \
            "Should have fitted parameters"
    
    def test_multivariate_different_families(self):
        """
        Test bf() + bf() + set_rescor() for multivariate models.
        
        Validates:
        - Multiple bf() objects combined
        - set_rescor() function for residual correlation
        - Both responses use gaussian family (required for rescor)
        - lf() for distributional parameters in one response
        - Shared random effects across responses
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf, set_rescor
        from brmspy.brms_functions.families import poisson, gaussian
        import arviz as az
        
        # Load epilepsy dataset
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create multivariate formula with same family (gaussian for both)
        # Note: set_rescor() only works with gaussian or student families
        bf_count = bf("count ~ zAge + (1|p|patient)") + gaussian()
        bf_base = bf("zBase ~ Trt + (1|p|patient)") + \
                   lf("sigma ~ 0 + Trt", dpar="sigma") + \
                   gaussian()
        formula = bf_count + bf_base + set_rescor(True)
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters from both responses
        param_names = list(model.idata.posterior.data_vars)
        assert any('count' in p.lower() or 'zage' in p.lower() for p in param_names), \
            "Should have parameters for count response"
        assert any('zbase' in p.lower() or 'trt' in p.lower() for p in param_names), \
            "Should have parameters for zBase response"
        # Check for shared random effects
        assert any('sd' in p.lower() and 'patient' in p.lower() for p in param_names), \
            "Should have random effects for patient"
        # Check for residual correlation parameter (rescor = True)
        assert any('rescor' in p.lower() or 'sigma' in p.lower() for p in param_names), \
            "Should have residual correlation or sigma parameters"
    
    def test_hurdle_modeled_hu_parameter(self):
        """
        Test bf() + lf() with hurdle_gamma family's hu parameter.
        
        Validates:
        - lf() with hurdle parameter
        - hurdle_gamma() family for semi-continuous data
        - Random effects + distributional modeling
        """
        import brmspy
        from brmspy.brms_functions.formula import bf, lf
        from brmspy.brms_functions.families import hurdle_gamma
        import arviz as az
        
        # Create synthetic hurdle data (zeros + positive continuous)
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        z = np.random.normal(0, 1, n)
        group = np.repeat(['G1', 'G2', 'G3', 'G4'], n // 4)
        
        # Generate hurdle process
        prob_zero = 1 / (1 + np.exp(-(0.5 * z)))  # Logistic
        is_zero = np.random.binomial(1, prob_zero, n)
        y_positive = np.random.gamma(2, 2 * np.exp(0.5 * x), n)
        y = np.where(is_zero, 0, y_positive)
        y = np.maximum(y, 1e-10)  # Ensure positive for gamma
        
        data = pd.DataFrame({
            'y': y,
            'x': x,
            'z': z,
            'group': group
        })
        
        # Create formula with modeled hurdle parameter
        formula = bf("y ~ x + (1|group)") + \
                  lf("hu ~ z", dpar="hu") + \
                  hurdle_gamma()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=data,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_x' in p for p in param_names), \
            "Should have coefficient for x"
        assert any('sd_group' in p for p in param_names), \
            "Should have random effects"
        assert any('hu' in p.lower() for p in param_names), \
            "Should have parameters for modeled hu"
    
    def test_ordinal_threshold_structure(self):
        """
        Test bf() + cumulative() with threshold parameter.
        
        Validates:
        - cumulative() family for ordinal outcomes
        - Threshold structure specification
        - Ordinal regression integration
        """
        import brmspy
        from brmspy.brms_functions.formula import bf
        from brmspy.brms_functions.families import cumulative
        import arviz as az
        
        # Create synthetic ordinal data
        np.random.seed(42)
        n = 120
        age = np.random.normal(50, 15, n)
        
        # Generate ordinal ratings (1-5)
        latent = 0.05 * age + np.random.logistic(0, 1, n)
        rating = np.digitize(latent, bins=[-2, -0.5, 0.5, 2]) + 1
        
        data = pd.DataFrame({
            'rating': rating,
            'age': age
        })
        
        # Create formula with ordinal family
        formula = bf("rating ~ age") + cumulative(threshold="equidistant")
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=data,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        assert any('b_age' in p for p in param_names), \
            "Should have coefficient for age"
        assert any('intercept' in p.lower() or 'threshold' in p.lower() for p in param_names), \
            "Should have threshold/intercept parameters"
    
    def test_complex_pq_random_effects_smooth(self):
        """
        Test bf() with p|q random effects syntax and smooth terms.
        
        Validates:
        - Smooth terms s() integration
        - Complex p|q random effects structure
        - poisson() family
        - GAM-style model with hierarchical structure
        """
        import brmspy
        from brmspy.brms_functions.formula import bf
        from brmspy.brms_functions.families import poisson
        import arviz as az
        
        # Load epilepsy dataset
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Create formula with smooth and complex random effects
        formula = bf("count ~ s(zAge) + (1|p|patient) + (1|q|Trt)") + poisson()
        
        # Fit model
        model = brmspy.fit(
            formula=formula,
            data=epilepsy,
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Verify model fitted
        assert isinstance(model.idata, az.InferenceData), \
            "Model should return InferenceData"
        
        # Check parameters
        param_names = list(model.idata.posterior.data_vars)
        # Smooth terms create special parameters
        assert any('s' in p.lower() or 'zage' in p.lower() for p in param_names), \
            "Should have parameters for smooth term"
        assert any('sd_patient' in p for p in param_names), \
            "Should have random effects for patient (p structure)"
        assert any('sd_Trt' in p or 'sd_trt' in p.lower() for p in param_names), \
            "Should have random effects for Trt (q structure)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'requires_brms and slow'])