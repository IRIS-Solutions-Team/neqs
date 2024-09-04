
import functools as _ft
from .iterative import iterate
from .quasi_newton import quasi_newton_step


quasi_newton = _ft.partial(iterate, step_eval=quasi_newton_step, )


import importlib.metadata as _md
__version__ = _md.version(__name__)
__doc__ = _md.metadata(__name__).json["description"]

