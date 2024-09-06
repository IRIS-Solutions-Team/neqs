"""
Quasi-Newton step
"""

import scipy as _sp
from .iterative import (
    FuncEvalType,
    JacobEvalType,
    NormEvalType,
    GuessType,
    ArrayType,
)


def quasi_newton_step(
    prev_guess: GuessType,
    prev_func: ArrayType,
    func_eval: FuncEvalType,
    jacob_eval: JacobEvalType,
    norm_eval: NormEvalType,
    *args,
) -> tuple[GuessType, bool]:
    """
    """

    # args[0] contains data
    jacob = jacob_eval(prev_guess, args[0])

    # calculate the unit step
    try:
        unit_step = _sp.sparse.linalg.spsolve(
            jacob,
            -prev_func,
        )
    except Exception:
        return prev_guess, False

    # optimize the step size
    # for now, not done, set to 1
    step_size = 1

    # calculate the new candidate
    new_guess = prev_guess + step_size * unit_step

    return new_guess, True
