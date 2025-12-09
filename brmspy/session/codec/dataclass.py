

from brmspy.session.codec.base import CodecRegistry, DataclassCodec



def register_dataclasses(registry: CodecRegistry):
    from brmspy.types import FitResult

    fit_result_codec = DataclassCodec(
        cls=FitResult,
        field_codecs={
            "r": "pickle",
            "idata": "arviz.InferenceData",
        },
        registry=registry,
    )
    registry.register(fit_result_codec)