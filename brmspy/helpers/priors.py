
import typing

from brmspy.helpers import singleton
from brmspy.types import PriorSpec


def _build_priors(priors: typing.Optional[typing.Sequence[PriorSpec]] = None) -> list:
    brms = singleton._get_brms()
    if not priors:
        return []

    prior_objs = []
    for p in priors:
        kwargs = p.to_brms_kwargs()
        # first argument is the prior string
        prior_str = kwargs.pop("prior")
        prior_obj = brms.prior_string(prior_str, **kwargs)
        prior_objs.append(prior_obj)

    brms_prior = prior_objs[0]
    for p in prior_objs[1:]:
        brms_prior = brms_prior + p

    assert brms.is_brmsprior(brms_prior)
    return brms_prior