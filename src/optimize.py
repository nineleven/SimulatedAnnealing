from dataclasses import dataclass

import numpy as np
import sympy

from src.strategy import AnnealingStrategy
from src.utils import CountCalls


@dataclass
class SimulatedAnnealingParameters:
    f: str
    x_0: np.ndarray
    n_runs: int
    max_calls: int
    T_max: float
    strategy: AnnealingStrategy
    max_step: float


@dataclass
class OptimizationResult:
    f_val: float
    x: np.ndarray


def make_objective(str_repr):
    expr = sympy.simplify(str_repr)
    x, y = sympy.symbols('x, y')

    @CountCalls
    def objective(p):
        return float(expr.evalf(subs={x: p[0], y: p[1]}))

    return objective


def sample_neighbor(x, params):
    step = np.random.uniform(-params.max_step, params.max_step, size=x.shape)
    return x + step


def simulated_annealing(params: SimulatedAnnealingParameters):
    T = params.T_max

    f = make_objective(params.f)
    x = params.x_0
    y = f(params.x_0)

    best_x = x
    best_y = y

    k = 1

    while T > 0 and f.n_calls < params.max_calls:
        n_x = sample_neighbor(x, params)
        n_y = f(n_x)

        dE = n_y - y

        if dE < 0 or np.random.rand() < np.exp(- dE / T):
            x, y = n_x, n_y

            if y < best_y:
                best_x, best_y = x, y

        k += 1
        T = params.strategy.update(T, k)

    return OptimizationResult(best_y, best_x)
