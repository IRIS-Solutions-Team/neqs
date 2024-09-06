"""
Quasi-Newton step
"""

import numpy as _np
import scipy as _sp
from .iterative import (
    FuncEvalType,
    JacobEvalType,
    NormEvalType,
    GuessType,
    ArrayType,
)


def backtracking_line_search(
    prev_guess: GuessType,
    prev_func: ArrayType,
    func_eval: FuncEvalType,
    norm_eval: NormEvalType,
    unit_step: ArrayType,
    args: dict,
    alpha: float = 1.0,
    beta_init: float = 0.8,
    beta_min: float = 0.1,
    beta_max: float = 0.9,
    c: float = 1e-4,
    max_iter: int = 1000,
) -> tuple[float, bool]:
    """
    Paper: https://ar5iv.labs.arxiv.org/html/1904.06321
    """

    # start off with an initial step size
    step_size = alpha
    beta = beta_init
    iter = 0

    # calculate initial norm of func
    prev_func_norm = norm_eval(prev_func)

    while iter <= max_iter:
        # check current step
        if iter == max_iter:
            return step_size, False

        # calculate the new candidate and evualuate it
        new_guess = prev_guess + step_size * unit_step
        new_func = func_eval(new_guess, args["data"])
        new_func_norm = norm_eval(new_func)

        # check the Armijo (sufficient decrease) condition
        # if not met, update the step size
        if new_func_norm <= prev_func_norm \
            + c * step_size * _np.dot(prev_func, unit_step):
            return step_size, True
        step_size *= beta

        # adjust beta dynamically based on function decrease
        if new_func_norm > prev_func_norm:
            # if function decrease is slow,
            # reduce beta to make larger reductions in step size
            beta = max(beta * 0.9, beta_min)
        else:
            # if function decrease is fast,
            # increase beta to avoid making small step size changes
            beta = min(beta * 1.1, beta_max)

        # prepare for the next iteration
        prev_func = new_func
        prev_func_norm = new_func_norm
        iter += 1


def quasi_newton_step(
    prev_guess: GuessType,
    prev_func: ArrayType,
    func_eval: FuncEvalType,
    jacob_eval: JacobEvalType,
    norm_eval: NormEvalType,
    args: dict,
) -> tuple[GuessType, bool]:
    """
    """

    # evaluate the Jacobian
    jacob = jacob_eval(prev_guess, args["data"])

    # calculate the unit step
    unit_step = _sp.sparse.linalg.spsolve(
        jacob,
        -prev_func,
    )

    # optimize the step
    # check the status as well
    step_size, status = backtracking_line_search(
        prev_guess,
        prev_func,
        func_eval,
        norm_eval,
        unit_step,
        args,
    )
    if not status:
        return prev_guess, False

    # calculate the new candidate
    new_guess = prev_guess + step_size * unit_step

    return new_guess, True
