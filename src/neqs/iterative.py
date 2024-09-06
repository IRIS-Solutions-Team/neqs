"""
Iterative numerical solver boilerplate
"""

import numpy as _np
import scipy as _sp
import enum as _en
import functools as _ft
from typing import (
    Any,
    Callable,
)
from typing import (
    TYPE_CHECKING, )
if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

STEP_TOL = 1e-4
FUNC_TOL = 1e-4
MAX_ITER = 1000
NORM_ORDER = None

ArrayType = _np.ndarray
SparseArrayType = _sp.sparse.spmatrix
FuncEvalType = Callable[[ArrayType], ArrayType]
JacobEvalType = Callable[[ArrayType], SparseArrayType]
NormEvalType = Callable[[ArrayType], float]
SettingsType = dict[str, Any]
GuessType = dict[str, Any]
StepEvalType = Callable[[GuessType, FuncEvalType, JacobEvalType, SettingsType],
                        ArrayType]


class ExitStatus(_en.Enum):
    """
    """

    CONVERGED = 0
    MAX_ITER = _en.auto()
    INVALID = _en.auto()


def check_convergence(
    guess,
    prev_guess,
    func,
    step_tolerance,
    func_tolerance,
    norm_eval,
) -> bool:
    """
    """

    # compute the norm of func
    norm_func = norm_eval(func)

    # compute the step size norm
    step_size = norm_eval(guess - prev_guess)

    # check the convergence conditions
    if norm_func < func_tolerance and step_size < step_tolerance:
        return True


def iterate(
    func_eval: FuncEvalType,
    jacob_eval: JacobEvalType,
    initial_guess: ArrayType,
    settings: SettingsType,
    args: dict,
    step_eval: StepEvalType,
) -> tuple[GuessType, ExitStatus]:
    """
    """

    guess = initial_guess
    prev_guess = guess

    # tolerance setting
    step_tolerance = settings.get("step_tolerance") \
        if settings.get("step_tolerance") is not None else STEP_TOL
    func_tolerance = settings.get("func_tolerance") \
        if settings.get("func_tolerance") is not None else FUNC_TOL

    # tolerance setting
    iter = 0
    max_iter = settings.get("max_iterations") \
        if settings.get("max_iterations") is not None else MAX_ITER

    # norm order setting
    norm_order = settings.get("norm_order") \
        if settings.get("norm_order") is not None else NORM_ORDER
    norm_eval = _ft.partial(
        _sp.linalg.norm,
        ord=norm_order,
    )

    while iter <= max_iter:
        # check current step
        if iter == max_iter:
            return guess, ExitStatus.MAX_ITER

        # evaluate the function
        func = func_eval(guess, args["data"])

        # check the convergence
        status = check_convergence(
            guess,
            prev_guess,
            func,
            step_tolerance,
            func_tolerance,
            norm_eval,
        )
        if status:
            return guess, ExitStatus.CONVERGED

        # save prev values for the next iteration
        prev_guess = guess
        prev_func = func
        iter += 1

        # calculate a new candidate
        guess, status = step_eval(
            prev_guess,
            prev_func,
            func_eval,
            jacob_eval,
            norm_eval,
            args,
        )

        if not status:
            return guess, ExitStatus.INVALID
