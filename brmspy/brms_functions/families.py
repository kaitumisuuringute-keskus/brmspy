"""

Reference:
[https://paulbuerkner.com/brms/reference/brmsfamily.html](https://paulbuerkner.com/brms/reference/brmsfamily.html)

family()
"""



from typing import Callable, Optional, Union, cast
from rpy2.robjects import ListVector
import rpy2.robjects as ro

from brmspy.helpers.conversion import kwargs_r
from brmspy.types import FitResult


def brmsfamily(
    family,
    link: Optional[str] = None,
    link_sigma: str = "log",
    link_shape: str = "log",
    link_nu: str = "logm1",
    link_phi: str = "log",
    link_kappa: str = "log",
    link_beta: str = "log",
    link_zi: str = "logit",
    link_hu: str = "logit",
    link_zoi: str = "logit",
    link_coi: str = "logit",
    link_disc: str = "log",
    link_bs: str = "log",
    link_ndt: str = "log",
    link_bias: str = "logit",
    link_xi: str = "log1p",
    link_alpha: str = "identity",
    link_quantile: str = "logit",
    threshold: str = "flexible",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """
    Family objects provide a convenient way to specify the details of the models used by many model fitting functions. The family functions presented here are for use with brms only and will **not** work with other model fitting functions such as glm or glmer. However, the standard family functions as described in family will work with brms.

    Parameters
    ----------
    family
        A character string naming the distribution family of the response variable to be used in the model. Currently, the following families are supported: gaussian, student, binomial, bernoulli, beta-binomial, poisson, negbinomial, geometric, Gamma, skew_normal, lognormal, shifted_lognormal, exgaussian, wiener, inverse.gaussian, exponential, weibull, frechet, Beta, dirichlet, von_mises, asym_laplace, gen_extreme_value, categorical, multinomial, cumulative, cratio, sratio, acat, hurdle_poisson, hurdle_negbinomial, hurdle_gamma, hurdle_lognormal, hurdle_cumulative, zero_inflated_binomial, zero_inflated_beta_binomial, zero_inflated_beta, zero_inflated_negbinomial, zero_inflated_poisson, and zero_one_inflated_beta.
    link
        A specification for the model link function. This can be a name/expression or character string. See the 'Details' section for more information on link functions supported by each family.
    link_sigma
        Link of auxiliary parameter sigma if being predicted.
    link_shape
        Link of auxiliary parameter shape if being predicted.
    link_nu
        Link of auxiliary parameter nu if being predicted.
    link_phi
        Link of auxiliary parameter phi if being predicted.
    link_kappa
        Link of auxiliary parameter kappa if being predicted.
    link_beta
        Link of auxiliary parameter beta if being predicted.
    link_zi
        Link of auxiliary parameter zi if being predicted.
    link_hu
        Link of auxiliary parameter hu if being predicted.
    link_zoi
        Link of auxiliary parameter zoi if being predicted.
    link_coi
        Link of auxiliary parameter coi if being predicted.
    link_disc
        Link of auxiliary parameter disc if being predicted.
    link_bs
        Link of auxiliary parameter bs if being predicted.
    link_ndt
        Link of auxiliary parameter ndt if being predicted.
    link_bias
        Link of auxiliary parameter bias if being predicted.
    link_xi
        Link of auxiliary parameter xi if being predicted.
    link_alpha
        Link of auxiliary parameter alpha if being predicted.
    link_quantile
        Link of auxiliary parameter quantile if being predicted.
    threshold
        A character string indicating the type of thresholds (i.e. intercepts) used in an ordinal model. "flexible" provides the standard unstructured thresholds, "equidistant" restricts the distance between consecutive thresholds to the same value, and "sum_to_zero" ensures the thresholds sum to zero.
    refcat
        Optional name of the reference response category used in categorical, multinomial, dirichlet and logistic_normal models. If NULL (the default), the first category is used as the reference. If NA, all categories will be predicted, which requires strong priors or carefully specified predictor terms in order to lead to an identified model.
    bhaz
        Currently for experimental purposes only.

    """
    r_brmsfamily = cast(Callable, ro.r('brms::brmsfamily'))

    collected_args = {
        "family": family,
        "link": link,
        "link_sigma": link_sigma,
        "link_shape": link_shape,
        "link_nu": link_nu,
        "link_phi": link_phi,
        "link_kappa": link_kappa,
        "link_beta": link_beta,
        "link_zi": link_zi,
        "link_hu": link_hu,
        "link_zoi": link_zoi,
        "link_coi": link_coi,
        "link_disc": link_disc,
        "link_bs": link_bs,
        "link_ndt": link_ndt,
        "link_bias": link_bias,
        "link_xi": link_xi,
        "link_alpha": link_alpha,
        "link_quantile": link_quantile,
        "threshold": threshold,
        "refcat": refcat,
        **kwargs
    }
    collected_args = kwargs_r(collected_args)

    return r_brmsfamily(**collected_args)

def family(fit: Union[FitResult, ListVector], **kwargs) -> ListVector:
    """Extract family object from a fitted model.
    
    Parameters
    ----------
    fit : FitResult or ListVector
        Fitted brms model
    """
    if isinstance(fit, FitResult):
        r_fit = fit.r
    else:
        r_fit = fit
    
    r_family = cast(Callable, ro.r('family'))
    kwargs = kwargs_r(kwargs)

    return r_family(
        r_fit, **kwargs
    )



def student(
    link: str = "identity",
    link_sigma: str = "log",
    link_nu: str = "logm1",
    **kwargs
) -> ListVector:
    """Student's t distribution for robust regression.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for sigma parameter
    link_nu : str
        Link function for degrees of freedom parameter
    """
    return brmsfamily(
        family="student",
        link=link,
        link_sigma=link_sigma,
        link_nu=link_nu,
        **kwargs,
    )


def bernoulli(
    link: str = "logit",
    **kwargs
) -> ListVector:
    """Bernoulli distribution for binary 0/1 outcomes.
    
    Parameters
    ----------
    link : str
        Link function for the probability parameter
    """
    return brmsfamily(
        family="bernoulli",
        link=link,
        **kwargs,
    )


def beta_binomial(
    link: str = "logit",
    link_phi: str = "log",
    **kwargs
) -> ListVector:
    """Beta-binomial distribution for overdispersed binomial data.
    
    Parameters
    ----------
    link : str
        Link function for the probability parameter
    link_phi : str
        Link function for the precision parameter
    """
    return brmsfamily(
        family="beta_binomial",
        link=link,
        link_phi=link_phi,
        **kwargs,
    )


def negbinomial(
    link: str = "log",
    link_shape: str = "log",
    **kwargs
) -> ListVector:
    """Negative binomial distribution for overdispersed count data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="negbinomial",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )


# not yet officially supported in brms
def negbinomial2(
    link: str = "log",
    link_sigma: str = "log",
    **kwargs
) -> ListVector:
    return brmsfamily(
        family="negbinomial2",
        link=link,
        link_sigma=link_sigma,
        **kwargs,
    )


def geometric(
    link: str = "log",
    **kwargs
) -> ListVector:
    """Geometric distribution for count data (negative binomial with shape=1).
    
    Parameters
    ----------
    link : str
        Link function for the mean
    """
    return brmsfamily(
        family="geometric",
        link=link,
        **kwargs,
    )


# do not export yet in brms
def discrete_weibull(
    link: str = "logit",
    link_shape: str = "log",
    **kwargs
) -> ListVector:
    return brmsfamily(
        family="discrete_weibull",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )


# do not export yet in brms
def com_poisson(
    link: str = "log",
    link_shape: str = "log",
    **kwargs
) -> ListVector:
    return brmsfamily(
        family="com_poisson",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )


def lognormal(
    link: str = "identity",
    link_sigma: str = "log",
    **kwargs
) -> ListVector:
    """Lognormal distribution for positive continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean on log scale
    link_sigma : str
        Link function for sigma parameter
    """
    return brmsfamily(
        family="lognormal",
        link=link,
        link_sigma=link_sigma,
        **kwargs,
    )


def shifted_lognormal(
    link: str = "identity",
    link_sigma: str = "log",
    link_ndt: str = "log",
    **kwargs
) -> ListVector:
    """Shifted lognormal distribution with non-decision time parameter.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for sigma parameter
    link_ndt : str
        Link function for non-decision time parameter
    """
    return brmsfamily(
        family="shifted_lognormal",
        link=link,
        link_sigma=link_sigma,
        link_ndt=link_ndt,
        **kwargs,
    )


def skew_normal(
    link: str = "identity",
    link_sigma: str = "log",
    link_alpha: str = "identity",
    **kwargs
) -> ListVector:
    """Skew normal distribution for asymmetric continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for sigma parameter
    link_alpha : str
        Link function for skewness parameter
    """
    return brmsfamily(
        family="skew_normal",
        link=link,
        link_sigma=link_sigma,
        link_alpha=link_alpha,
        **kwargs,
    )


def exponential(
    link: str = "log",
    **kwargs
) -> ListVector:
    """Exponential distribution for time-to-event data.
    
    Parameters
    ----------
    link : str
        Link function for the rate parameter
    """
    return brmsfamily(
        family="exponential",
        link=link,
        **kwargs,
    )


def weibull(
    link: str = "log",
    link_shape: str = "log",
    **kwargs
) -> ListVector:
    """Weibull distribution for survival and reliability analysis.
    
    Parameters
    ----------
    link : str
        Link function for the scale parameter
    link_shape : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="weibull",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )


def frechet(
    link: str = "log",
    link_nu: str = "logm1",
    **kwargs
) -> ListVector:
    """Frechet distribution for extreme value analysis.
    
    Parameters
    ----------
    link : str
        Link function for the scale parameter
    link_nu : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="frechet",
        link=link,
        link_nu=link_nu,
        **kwargs,
    )


def gen_extreme_value(
    link: str = "identity",
    link_sigma: str = "log",
    link_xi: str = "log1p",
    **kwargs
) -> ListVector:
    """Generalized extreme value distribution for extreme events.
    
    Parameters
    ----------
    link : str
        Link function for the location parameter
    link_sigma : str
        Link function for the scale parameter
    link_xi : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="gen_extreme_value",
        link=link,
        link_sigma=link_sigma,
        link_xi=link_xi,
        **kwargs,
    )


def exgaussian(
    link: str = "identity",
    link_sigma: str = "log",
    link_beta: str = "log",
    **kwargs
) -> ListVector:
    """Ex-Gaussian distribution for reaction time data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for Gaussian SD parameter
    link_beta : str
        Link function for exponential rate parameter
    """
    return brmsfamily(
        family="exgaussian",
        link=link,
        link_sigma=link_sigma,
        link_beta=link_beta,
        **kwargs,
    )


def wiener(
    link: str = "identity",
    link_bs: str = "log",
    link_ndt: str = "log",
    link_bias: str = "logit",
    **kwargs
) -> ListVector:
    """Wiener diffusion model for two-choice reaction time data.
    
    Parameters
    ----------
    link : str
        Link function for drift rate
    link_bs : str
        Link function for boundary separation
    link_ndt : str
        Link function for non-decision time
    link_bias : str
        Link function for initial bias
    """
    return brmsfamily(
        family="wiener",
        link=link,
        link_bs=link_bs,
        link_ndt=link_ndt,
        link_bias=link_bias,
        **kwargs,
    )


def Beta(
    link: str = "logit",
    link_phi: str = "log",
    **kwargs
) -> ListVector:
    """Beta distribution for data between 0 and 1.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_phi : str
        Link function for the precision parameter
    """
    return brmsfamily(
        family="beta",
        link=link,
        link_phi=link_phi,
        **kwargs,
    )


def xbeta(
    link: str = "logit",
    link_phi: str = "log",
    link_kappa: str = "log",
    **kwargs
) -> ListVector:
    """Extended beta distribution with additional shape parameter.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_phi : str
        Link function for precision parameter
    link_kappa : str
        Link function for kappa shape parameter
    """
    return brmsfamily(
        family="xbeta",
        link=link,
        link_phi=link_phi,
        link_kappa=link_kappa,
        **kwargs,
    )


def dirichlet(
    link: str = "logit",
    link_phi: str = "log",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """Dirichlet distribution for compositional data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_phi : str
        Link function for the precision parameter
    refcat : str, optional
        Reference category
    """
    return brmsfamily(
        family="dirichlet",
        link=link,
        link_phi=link_phi,
        refcat=refcat,
        **kwargs,
    )


# not yet exported in brms
def dirichlet2(
    link: str = "log",
    # NOTE: R version uses refcat = NA; here default None
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    return brmsfamily(
        family="dirichlet2",
        link=link,
        refcat=refcat,
        **kwargs,
    )


def logistic_normal(
    link: str = "identity",
    link_sigma: str = "log",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """Logistic-normal distribution for compositional data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for sigma parameter
    refcat : str, optional
        Reference category
    """
    return brmsfamily(
        family="logistic_normal",
        link=link,
        link_sigma=link_sigma,
        refcat=refcat,
        **kwargs,
    )


def von_mises(
    link: str = "tan_half",
    link_kappa: str = "log",
    **kwargs
) -> ListVector:
    """Von Mises distribution for circular/directional data.
    
    Parameters
    ----------
    link : str
        Link function for the mean direction
    link_kappa : str
        Link function for concentration parameter
    """
    return brmsfamily(
        family="von_mises",
        link=link,
        link_kappa=link_kappa,
        **kwargs,
    )


def asym_laplace(
    link: str = "identity",
    link_sigma: str = "log",
    link_quantile: str = "logit",
    **kwargs
) -> ListVector:
    """Asymmetric Laplace distribution for quantile regression.
    
    Parameters
    ----------
    link : str
        Link function for the location
    link_sigma : str
        Link function for sigma parameter
    link_quantile : str
        Link function for the quantile parameter
    """
    return brmsfamily(
        family="asym_laplace",
        link=link,
        link_sigma=link_sigma,
        link_quantile=link_quantile,
        **kwargs,
    )


# do not export yet in brms
def zero_inflated_asym_laplace(
    link: str = "identity",
    link_sigma: str = "log",
    link_quantile: str = "logit",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    return brmsfamily(
        family="zero_inflated_asym_laplace",
        link=link,
        link_sigma=link_sigma,
        link_quantile=link_quantile,
        link_zi=link_zi,
        **kwargs,
    )


def cox(
    link: str = "log",
    **kwargs
) -> ListVector:
    """Cox proportional hazards model for survival data.
    
    Parameters
    ----------
    link : str
        Link function for the hazard rate
    """
    # original R wrapper doesn't pass slink; brmsfamily doesn't need it
    return brmsfamily(
        family="cox",
        link=link,
        **kwargs,
    )


def hurdle_poisson(
    link: str = "log",
    link_hu: str = "logit",
    **kwargs
) -> ListVector:
    """Hurdle Poisson distribution for zero-inflated count data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_hu : str
        Link function for hurdle parameter
    """
    return brmsfamily(
        family="hurdle_poisson",
        link=link,
        link_hu=link_hu,
        **kwargs,
    )


def hurdle_negbinomial(
    link: str = "log",
    link_shape: str = "log",
    link_hu: str = "logit",
    **kwargs
) -> ListVector:
    """Hurdle negative binomial for overdispersed zero-inflated count data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for shape parameter
    link_hu : str
        Link function for hurdle parameter
    """
    return brmsfamily(
        family="hurdle_negbinomial",
        link=link,
        link_shape=link_shape,
        link_hu=link_hu,
        **kwargs,
    )


def hurdle_gamma(
    link: str = "log",
    link_shape: str = "log",
    link_hu: str = "logit",
    **kwargs
) -> ListVector:
    """Hurdle Gamma distribution for zero-inflated positive continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for shape parameter
    link_hu : str
        Link function for hurdle parameter
    """
    return brmsfamily(
        family="hurdle_gamma",
        link=link,
        link_shape=link_shape,
        link_hu=link_hu,
        **kwargs,
    )


def hurdle_lognormal(
    link: str = "identity",
    link_sigma: str = "log",
    link_hu: str = "logit",
    **kwargs
) -> ListVector:
    """Hurdle lognormal for zero-inflated positive continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for sigma parameter
    link_hu : str
        Link function for hurdle parameter
    """
    return brmsfamily(
        family="hurdle_lognormal",
        link=link,
        link_sigma=link_sigma,
        link_hu=link_hu,
        **kwargs,
    )


def hurdle_cumulative(
    link: str = "logit",
    link_hu: str = "logit",
    link_disc: str = "log",
    threshold: str = "flexible",
    **kwargs
) -> ListVector:
    """Hurdle cumulative for zero-inflated ordinal data.
    
    Parameters
    ----------
    link : str
        Link function for the ordinal response
    link_hu : str
        Link function for hurdle parameter
    link_disc : str
        Link function for discrimination parameter
    threshold : str
        Type of threshold structure
    """
    return brmsfamily(
        family="hurdle_cumulative",
        link=link,
        link_hu=link_hu,
        link_disc=link_disc,
        threshold=threshold,
        **kwargs,
    )


def zero_inflated_beta(
    link: str = "logit",
    link_phi: str = "log",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-inflated beta for data between 0 and 1 with excess zeros.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_phi : str
        Link function for precision parameter
    link_zi : str
        Link function for zero-inflation parameter
    """
    return brmsfamily(
        family="zero_inflated_beta",
        link=link,
        link_phi=link_phi,
        link_zi=link_zi,
        **kwargs,
    )


def zero_one_inflated_beta(
    link: str = "logit",
    link_phi: str = "log",
    link_zoi: str = "logit",
    link_coi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-one-inflated beta for data between 0 and 1 with excess zeros and ones.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_phi : str
        Link function for precision parameter
    link_zoi : str
        Link function for zero-or-one inflation parameter
    link_coi : str
        Link function for conditional one inflation parameter
    """
    return brmsfamily(
        family="zero_one_inflated_beta",
        link=link,
        link_phi=link_phi,
        link_zoi=link_zoi,
        link_coi=link_coi,
        **kwargs,
    )


def zero_inflated_poisson(
    link: str = "log",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-inflated Poisson for count data with excess zeros.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_zi : str
        Link function for zero-inflation parameter
    """
    return brmsfamily(
        family="zero_inflated_poisson",
        link=link,
        link_zi=link_zi,
        **kwargs,
    )


def zero_inflated_negbinomial(
    link: str = "log",
    link_shape: str = "log",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-inflated negative binomial for overdispersed count data with excess zeros.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for shape parameter
    link_zi : str
        Link function for zero-inflation parameter
    """
    return brmsfamily(
        family="zero_inflated_negbinomial",
        link=link,
        link_shape=link_shape,
        link_zi=link_zi,
        **kwargs,
    )


def zero_inflated_binomial(
    link: str = "logit",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-inflated binomial for binary count data with excess zeros.
    
    Parameters
    ----------
    link : str
        Link function for probability parameter
    link_zi : str
        Link function for zero-inflation parameter
    """
    return brmsfamily(
        family="zero_inflated_binomial",
        link=link,
        link_zi=link_zi,
        **kwargs,
    )


def zero_inflated_beta_binomial(
    link: str = "logit",
    link_phi: str = "log",
    link_zi: str = "logit",
    **kwargs
) -> ListVector:
    """Zero-inflated beta-binomial for overdispersed binomial data with excess zeros.
    
    Parameters
    ----------
    link : str
        Link function for probability parameter
    link_phi : str
        Link function for precision parameter
    link_zi : str
        Link function for zero-inflation parameter
    """
    return brmsfamily(
        family="zero_inflated_beta_binomial",
        link=link,
        link_phi=link_phi,
        link_zi=link_zi,
        **kwargs,
    )


def categorical(
    link: str = "logit",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """Categorical distribution for unordered multi-category outcomes.
    
    Parameters
    ----------
    link : str
        Link function for category probabilities
    refcat : str, optional
        Reference category
    """
    return brmsfamily(
        family="categorical",
        link=link,
        refcat=refcat,
        **kwargs,
    )


def multinomial(
    link: str = "logit",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """Multinomial distribution for count data across multiple categories.
    
    Parameters
    ----------
    link : str
        Link function for category probabilities
    refcat : str, optional
        Reference category
    """
    return brmsfamily(
        family="multinomial",
        link=link,
        refcat=refcat,
        **kwargs,
    )


def dirichlet_multinomial(
    link: str = "logit",
    link_phi: str = "log",
    refcat: Optional[str] = None,
    **kwargs
) -> ListVector:
    """Dirichlet-multinomial for overdispersed categorical count data.
    
    Parameters
    ----------
    link : str
        Link function for category probabilities
    link_phi : str
        Link function for precision parameter
    refcat : str, optional
        Reference category
    """
    return brmsfamily(
        family="dirichlet_multinomial",
        link=link,
        link_phi=link_phi,
        refcat=refcat,
        **kwargs,
    )


def cumulative(
    link: str = "logit",
    link_disc: str = "log",
    threshold: str = "flexible",
    **kwargs
) -> ListVector:
    """Cumulative (proportional odds) model for ordinal outcomes.
    
    Parameters
    ----------
    link : str
        Link function for cumulative probabilities
    link_disc : str
        Link function for discrimination parameter
    threshold : str
        Type of threshold structure
    """
    return brmsfamily(
        family="cumulative",
        link=link,
        link_disc=link_disc,
        threshold=threshold,
        **kwargs,
    )


def sratio(
    link: str = "logit",
    link_disc: str = "log",
    threshold: str = "flexible",
    **kwargs
) -> ListVector:
    """Sequential (stopping) ratio model for ordinal outcomes.
    
    Parameters
    ----------
    link : str
        Link function for sequential ratios
    link_disc : str
        Link function for discrimination parameter
    threshold : str
        Type of threshold structure
    """
    return brmsfamily(
        family="sratio",
        link=link,
        link_disc=link_disc,
        threshold=threshold,
        **kwargs,
    )


def cratio(
    link: str = "logit",
    link_disc: str = "log",
    threshold: str = "flexible",
    **kwargs
) -> ListVector:
    """Continuation ratio model for ordinal outcomes.
    
    Parameters
    ----------
    link : str
        Link function for continuation ratios
    link_disc : str
        Link function for discrimination parameter
    threshold : str
        Type of threshold structure
    """
    return brmsfamily(
        family="cratio",
        link=link,
        link_disc=link_disc,
        threshold=threshold,
        **kwargs,
    )


def acat(
    link: str = "logit",
    link_disc: str = "log",
    threshold: str = "flexible",
    **kwargs
) -> ListVector:
    """Adjacent category model for ordinal outcomes.
    
    Parameters
    ----------
    link : str
        Link function for adjacent category ratios
    link_disc : str
        Link function for discrimination parameter
    threshold : str
        Type of threshold structure
    """
    return brmsfamily(
        family="acat",
        link=link,
        link_disc=link_disc,
        threshold=threshold,
        **kwargs,
    )

def gaussian(
    link: str = "identity",
    link_sigma: str = "log",
    **kwargs,
) -> ListVector:
    """Gaussian (normal) distribution for continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_sigma : str
        Link function for the standard deviation
    """
    return brmsfamily(
        family="gaussian",
        link=link,
        link_sigma=link_sigma,
        **kwargs,
    )


def poisson(
    link: str = "log",
    **kwargs,
) -> ListVector:
    """Poisson distribution for count data.
    
    Parameters
    ----------
    link : str
        Link function for the rate parameter
    """
    return brmsfamily(
        family="poisson",
        link=link,
        **kwargs,
    )


def binomial(
    link: str = "logit",
    **kwargs,
) -> ListVector:
    """Binomial distribution for binary count data.
    
    Parameters
    ----------
    link : str
        Link function for the probability parameter
    """
    return brmsfamily(
        family="binomial",
        link=link,
        **kwargs,
    )


def Gamma(
    link: str = "log",
    link_shape: str = "log",
    **kwargs,
) -> ListVector:
    """Gamma distribution for positive continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="Gamma",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )


def inverse_gaussian(
    link: str = "1/mu^2",
    link_shape: str = "log",
    **kwargs,
) -> ListVector:
    """Inverse Gaussian distribution for positive continuous data.
    
    Parameters
    ----------
    link : str
        Link function for the mean
    link_shape : str
        Link function for the shape parameter
    """
    return brmsfamily(
        family="inverse.gaussian",
        link=link,
        link_shape=link_shape,
        **kwargs,
    )