import irispie as ir
from neqs import *

# load the pickled data
pickle = ir.load("tests/evaluator.pkl")
evaluator = pickle["evaluator"]
data = pickle["data"]

test, status = quasi_newton(
    evaluator.eval_func,
    evaluator.eval_jacob,
    evaluator.get_init_guess(data),
    {
        "step_tolerance": 0.01,
        "func_tolerance": 0.01,
        "max_iterations": 1000,
        "norm_order": None,
    },
    data,
)

print(f"Solver status: {status}")
print(f"Solver results: {test}")
