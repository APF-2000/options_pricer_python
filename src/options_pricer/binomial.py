"""Cox-Ross-Rubinstein binomial pricing for European and American options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from time import perf_counter

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .instruments import OptionContract, OptionType
from .validation import ensure_option_type


@dataclass(frozen=True)
class BinomialResult:
    """Binomial pricing result."""

    price: float
    steps: int
    up_factor: float
    down_factor: float
    risk_neutral_probability: float
    exercise_style: str
    early_exercise_count: int = 0
    stock_tree: list[list[float]] | None = None
    option_tree: list[list[float]] | None = None
    early_exercise_tree: list[list[bool]] | None = None
    elapsed_seconds: float | None = None


def _intrinsic_vector(stock_prices: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(stock_prices - strike, 0.0)
    return np.maximum(strike - stock_prices, 0.0)


def _deterministic_binomial(
    option: OptionContract,
    option_type: OptionType,
    *,
    steps: int,
    american: bool,
    return_tree: bool,
) -> BinomialResult:
    dt = option.time_to_maturity / steps
    growth = exp((option.rate - option.dividend_yield) * dt)
    discount = exp(-option.rate * dt)
    stock_levels = [[option.spot * (growth**level)] for level in range(steps + 1)]
    values = np.array(
        [_intrinsic_vector(np.array([stock_levels[-1][0]]), option.strike, option_type)[0]],
        dtype=float,
    )
    option_tree = [[values[0]]] if return_tree else None
    exercise_tree = [[False]] if return_tree else None
    early_count = 0

    for level in range(steps - 1, -1, -1):
        continuation = discount * values[0]
        stock = stock_levels[level][0]
        exercise = option.intrinsic_value(option_type, spot=stock)
        if american and exercise > continuation + 1e-14:
            value = exercise
            exercised = True
            early_count += 1
        else:
            value = continuation
            exercised = False
        values = np.array([value], dtype=float)
        if return_tree and option_tree is not None and exercise_tree is not None:
            option_tree.insert(0, [value])
            exercise_tree.insert(0, [exercised])

    return BinomialResult(
        price=float(values[0]),
        steps=steps,
        up_factor=1.0,
        down_factor=1.0,
        risk_neutral_probability=1.0,
        exercise_style="american" if american else "european",
        early_exercise_count=early_count,
        stock_tree=stock_levels if return_tree else None,
        option_tree=option_tree,
        early_exercise_tree=exercise_tree,
    )


def cox_ross_rubinstein_price(
    option: OptionContract,
    option_type: OptionType,
    *,
    steps: int = 100,
    american: bool | None = None,
    return_tree: bool = False,
) -> BinomialResult:
    ensure_option_type(option_type)
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    use_american = option.exercise_style == "american" if american is None else american
    tau = option.time_to_maturity
    start = perf_counter()

    if tau == 0.0:
        payoff = option.intrinsic_value(option_type)
        elapsed = perf_counter() - start
        return BinomialResult(
            price=payoff,
            steps=steps,
            up_factor=1.0,
            down_factor=1.0,
            risk_neutral_probability=0.5,
            exercise_style="american" if use_american else "european",
            stock_tree=[[option.spot]] if return_tree else None,
            option_tree=[[payoff]] if return_tree else None,
            early_exercise_tree=[[False]] if return_tree else None,
            elapsed_seconds=elapsed,
        )

    if option.volatility == 0.0:
        result = _deterministic_binomial(
            option,
            option_type,
            steps=steps,
            american=use_american,
            return_tree=return_tree,
        )
        return BinomialResult(**{**result.__dict__, "elapsed_seconds": perf_counter() - start})

    dt = tau / steps
    up_factor = exp(option.volatility * sqrt(dt))
    down_factor = 1.0 / up_factor
    growth = exp((option.rate - option.dividend_yield) * dt)
    probability = (growth - down_factor) / (up_factor - down_factor)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "risk-neutral probability is outside [0, 1]; increase the number of steps"
        )

    up_powers = np.arange(steps, -1, -1, dtype=float)
    down_powers = np.arange(0, steps + 1, dtype=float)
    stock_terminal = option.spot * (up_factor**up_powers) * (down_factor**down_powers)
    option_values = _intrinsic_vector(stock_terminal, option.strike, option_type)

    stock_tree: list[list[float]] | None = None
    option_tree: list[list[float]] | None = None
    exercise_tree: list[list[bool]] | None = None
    early_count = 0
    if return_tree:
        stock_tree = [stock_terminal.tolist()]
        option_tree = [option_values.tolist()]
        exercise_tree = [[False] * (steps + 1)]

    discount = exp(-option.rate * dt)
    for level in range(steps - 1, -1, -1):
        option_values = discount * (
            probability * option_values[:-1] + (1.0 - probability) * option_values[1:]
        )
        exercised = np.zeros(level + 1, dtype=bool)
        stock_values = option.spot * (
            up_factor ** np.arange(level, -1, -1, dtype=float)
        ) * (down_factor ** np.arange(0, level + 1, dtype=float))
        if use_american:
            exercise_values = _intrinsic_vector(stock_values, option.strike, option_type)
            exercised = exercise_values > option_values + 1e-14
            early_count += int(exercised.sum())
            option_values = np.maximum(option_values, exercise_values)
        if return_tree and stock_tree is not None and option_tree is not None and exercise_tree is not None:
            stock_tree.insert(0, stock_values.tolist())
            option_tree.insert(0, option_values.tolist())
            exercise_tree.insert(0, exercised.tolist())

    return BinomialResult(
        price=float(option_values[0]),
        steps=steps,
        up_factor=up_factor,
        down_factor=down_factor,
        risk_neutral_probability=probability,
        exercise_style="american" if use_american else "european",
        early_exercise_count=early_count,
        stock_tree=stock_tree,
        option_tree=option_tree,
        early_exercise_tree=exercise_tree,
        elapsed_seconds=perf_counter() - start,
    )


def binomial_convergence_table(
    option: OptionContract,
    option_type: OptionType,
    *,
    step_grid: list[int] | tuple[int, ...] = (5, 10, 25, 50, 100, 200, 400),
) -> pd.DataFrame:
    benchmark = black_scholes_price(option, option_type)
    rows: list[dict[str, float]] = []
    for steps in step_grid:
        result = cox_ross_rubinstein_price(option, option_type, steps=steps)
        rows.append(
            {
                "steps": float(steps),
                "binomial_price": result.price,
                "black_scholes_price": benchmark,
                "abs_error": abs(result.price - benchmark),
                "runtime_seconds": result.elapsed_seconds or 0.0,
            }
        )
    return pd.DataFrame(rows)
