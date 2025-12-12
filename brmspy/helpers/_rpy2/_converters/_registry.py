from ....types.rpy2_converters import Py2rConverter, R2pyConverter

_R2PY_CONVERTERS: dict[type | tuple[type, ...], R2pyConverter] = {}
_PY2R_CONVERTERS: dict[type | tuple[type, ...], Py2rConverter] = {}
