"""
Iterative numerical solver boilerplate
"""


#[
import numpy as _np
import scipy as _sp
import enum as _en

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Callable, )
    from collections.abc import (Iterable, Sequence, )
#]


ArrayType = _np.ndarray
SparseArrayType = _sp.sparse.spmatrix
StepEvalType = Callable[[GuessType, FuncEvalType, JacobEvalType, SettingsType], ArrayType]
FuncEvalType = Callable[[ArrayType], ArrayType]
JacobEvalType = Callable[[ArrayType], SparseArrayType]
NormEvalType = Callable[[ArrayType], float]
Settings = dict[str, Any]
GuessType = dict[str, Any]


class ExitStatus(_en.Enum):
    """
    """
    #[

    CONVERGED = 0
    MAX_ITER = _en.auto()
    INVALID = _en.auto()

    #]


def iterate(
    func_eval: FuncEvalType,
    jacob_eval: JacobEvalType,
    settings: SettingsType,
    *,
    step_eval: StepEvalType,
) -> tuple[GuessType, InfoType]:
    """
    """
    #[

    norm_eval = _ft.partial(_sp.linalg.norm, ord=settings["norm_order"], )

    #]


