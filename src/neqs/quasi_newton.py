"""
Quasi-Newton step
"""


#[
import numpy as _np
import scipy as _sp

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from .iterator import (
        FuncEvalType, JacobEvalType, NormEvalType,
        SettingsType, GuessType,
    )
#]


def quasi_newton_step(
    prev_guess: GuessType,
    prev_func: ArrayType,
    func_eval: FuncEvalType,
    jacob_eval: JacobEvalType,
    norm_eval: NormEvalType,
) -> GuessType:

    ...

    return new_guess

