"""
"""


#[

import importlib.metadata as _md
import functools as _ft

from .iterative import iterate, ExitStatus
from . import damped_newton
from .iter_printers import IterPrinter

#]


damped_newton = _ft.partial(iterate, damped_newton.eval_step, )


__version__ = _md.version(__name__)
__doc__ = _md.metadata(__name__).json["description"]

