

import importlib.metadata as _md


__version__ = _md.version(__name__)
__doc__ = _md.metadata(__name__).json["description"]


from .quasi_newton import *
from .quasi_newton import __all__ as quasi_newton_all


__all__ = (
    *quasi_newton_all,
)


