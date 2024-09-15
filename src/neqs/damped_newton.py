"""
Damped Newton step
"""


#[

import numpy as _np
import scipy as _sp
from numbers import Real

from . import iterative as _iterative

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from . import iterative as _iterative

#]


_DAMPING_FACTOR = 0.9
_MIN_STEP = 0.05
_MIN_IMPROVEMENT = 0


def eval_step(
    guess: _iterative.GuessType,
    func: _iterative.ArrayType,
    jacob: _iterative.SparseArrayType,
    norm: float,
    eval_func: _iterative.FuncEvalType,
    eval_norm: _iterative.NormEvalType,
) -> tuple[_iterative.GuessType, _iterative.ArrayType, Real, ]:
    """
    """

    direction = - _sp.sparse.linalg.spsolve(jacob, func, )

    new_guess, new_func, new_step, = _damp_search(
        direction=direction,
        prev_guess=guess,
        prev_norm=norm,
        eval_func=eval_func,
        eval_norm=eval_norm,
    )

    return new_guess, new_func, new_step


def _damp_search(
    direction: _iterative.ArrayType,
    prev_guess: _iterative.GuessType,
    prev_norm: _iterative.ArrayType,
    eval_func: _iterative.FuncEvalType,
    eval_norm: _iterative.NormEvalType,
    ) -> tuple[_iterative.GuessType, _iterative.ArrayType, Real, ]:
    """
    """
    min_norm = (1 - _MIN_IMPROVEMENT) * prev_norm
    new_step = None

    while True:
        new_step = 1 if new_step is None else (_DAMPING_FACTOR * new_step)
        if new_step < _MIN_STEP:
            raise _iterative.StepFailure
        new_guess = prev_guess + new_step * direction
        new_func = eval_func(new_guess, )
        new_norm = eval_norm(new_func, )
        if new_norm < min_norm:
            break

    return new_guess, new_func, new_step,


# def _backtracking_line_search(
#     prev_guess: GuessType,
#     prev_func: ArrayType,
#     eval_func: FuncEvalType,
#     eval_norm: NormEvalType,
#     unit_step: ArrayType,
#     args: dict,
#     alpha: float = 1.0,
#     beta_init: float = 0.8,
#     beta_min: float = 0.1,
#     beta_max: float = 0.9,
#     c: float = 1e-4,
#     max_iter: int = 1000,
# ) -> tuple[float, bool]:
#     """
#     Paper: https://ar5iv.labs.arxiv.org/html/1904.06321
#     """
# 
#     # start off with an initial step size
#     step_size = alpha
#     beta = beta_init
#     iter = 0
# 
#     # define epsilon for the gradient approximation
#     epsilon = _np.sqrt(_np.finfo(float).eps)
# 
#     # calculate the initial norm of func
#     prev_func_norm = eval_norm(prev_func)
# 
#     while True:
#         # check current step
#         if iter == max_iter:
#             return step_size, False
# 
#         # calculate the new candidate and evualuate it
#         new_guess = prev_guess + step_size * unit_step
#         new_func = eval_func(new_guess, args["data"])
# 
#         # calculate the gradient
#         prev_grad = _sp.optimize.approx_fprime(
#             prev_guess,
#             eval_func,
#             epsilon,
#             args["data"],
#         )
# 
#         # calculate the decrease
#         decrease = c * step_size * _np.dot(prev_grad, unit_step)
# 
#         # calculate the norms
#         new_func_norm = eval_norm(new_func)
#         decrease_norm = eval_norm(decrease)
# 
#         # check the Armijo (sufficient decrease) condition
#         # if not met, update the step size
#         if new_func_norm <= prev_func_norm + decrease_norm:
#             return step_size, True
#         step_size *= beta
# 
#         # adjust beta dynamically based on function decrease
#         if new_func_norm > prev_func_norm:
#             # if function decrease is slow,
#             # reduce beta to make larger reductions in step size
#             beta = max(beta * 0.9, beta_min)
#         else:
#             # if function decrease is fast,
#             # increase beta to avoid making small step size changes
#             beta = min(beta * 1.1, beta_max)
# 
#         # prepare for the next iteration
#         prev_func_norm = new_func_norm
#         iter += 1
