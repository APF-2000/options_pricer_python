"""Vectorised Monte Carlo pricing and convergence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from time import perf_counter

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .instruments import OptionContract, OptionType
from .validation import ensure_option_type


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo estimator output."""

    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    discounted_payoffs: np.ndarray
    terminal_prices: np.ndarray
    sample_paths: np.ndarray
    elapsed_seconds: float
    method: str
    control_variate_beta: float | None = None


def simulate_gbm_paths(
    option: OptionContract,
    *,
    paths: int,
    steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    drift: float | None = None,
) -> np.ndarray:
    if paths <= 0:
        raise ValueError("paths must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    generator = np.random.default_rng(seed)
    half_paths = (paths + 1) // 2 if antithetic else paths
    shocks = generator.standard_normal((half_paths, steps))
    if antithetic:
        shocks = np.vstack([shocks, -shocks])[:paths]

    dt = option.time_to_maturity / steps
    realised_drift = option.rate - option.dividend_yield if drift is None else drift
    increments = (
        (realised_drift - 0.5 * option.volatility**2) * dt
        + option.volatility * np.sqrt(dt) * shocks
    )
    log_paths = np.cumsum(increments, axis=1)
    spots = option.spot * np.exp(log_paths)
    return np.column_stack([np.full(paths, option.spot), spots])


def _discounted_payoff(
    option: OptionContract,
    option_type: OptionType,
    terminal_prices: np.ndarray,
) -> np.ndarray:
    if option_type == "call":
        payoff = np.maximum(terminal_prices - option.strike, 0.0)
    else:
        payoff = np.maximum(option.strike - terminal_prices, 0.0)
    return np.exp(-option.rate * option.time_to_maturity) * payoff


def monte_carlo_price(
    option: OptionContract,
    option_type: OptionType,
    *,
    paths: int = 10_000,
    steps: int = 252,
    seed: int | None = 7,
    antithetic: bool = True,
    control_variate: bool = False,
    sample_path_count: int = 8,
) -> MonteCarloResult:
    ensure_option_type(option_type)
    start = perf_counter()
    tau = option.time_to_maturity
    if paths <= 0:
        raise ValueError("paths must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    if tau == 0.0:
        payoff = option.intrinsic_value(option_type)
        elapsed = perf_counter() - start
        return MonteCarloResult(
            price=payoff,
            std_error=0.0,
            confidence_interval=(payoff, payoff),
            discounted_payoffs=np.array([payoff]),
            terminal_prices=np.array([option.spot]),
            sample_paths=np.array([[option.spot]]),
            elapsed_seconds=elapsed,
            method="deterministic",
        )

    paths_matrix = simulate_gbm_paths(
        option,
        paths=paths,
        steps=steps,
        seed=seed,
        antithetic=antithetic,
    )
    terminal_prices = paths_matrix[:, -1]
    discounted_payoffs = _discounted_payoff(option, option_type, terminal_prices)
    method = "antithetic" if antithetic else "standard"
    beta: float | None = None

    if control_variate:
        discounted_terminal_stock = np.exp(-option.rate * tau) * terminal_prices
        expected_discounted_terminal = option.spot * np.exp(-option.dividend_yield * tau)
        covariance = np.cov(discounted_payoffs, discounted_terminal_stock, ddof=1)
        variance_control = float(covariance[1, 1])
        beta = 0.0 if variance_control == 0.0 else float(covariance[0, 1] / variance_control)
        discounted_payoffs = discounted_payoffs - beta * (
            discounted_terminal_stock - expected_discounted_terminal
        )
        method = f"{method}+control_variate"

    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs)))
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return MonteCarloResult(
        price=price,
        std_error=std_error,
        confidence_interval=ci,
        discounted_payoffs=discounted_payoffs,
        terminal_prices=terminal_prices,
        sample_paths=paths_matrix[:sample_path_count],
        elapsed_seconds=perf_counter() - start,
        method=method,
        control_variate_beta=beta,
    )


def compare_monte_carlo_methods(
    option: OptionContract,
    option_type: OptionType,
    *,
    paths: int = 10_000,
    steps: int = 252,
    seed: int = 7,
) -> pd.DataFrame:
    benchmark = black_scholes_price(option, option_type)
    rows: list[dict[str, float | str]] = []
    for antithetic, control_variate in (
        (False, False),
        (True, False),
        (True, True),
    ):
        result = monte_carlo_price(
            option,
            option_type,
            paths=paths,
            steps=steps,
            seed=seed,
            antithetic=antithetic,
            control_variate=control_variate,
        )
        rows.append(
            {
                "method": result.method,
                "price": result.price,
                "abs_error": abs(result.price - benchmark),
                "std_error": result.std_error,
                "ci_width": result.confidence_interval[1] - result.confidence_interval[0],
                "runtime_seconds": result.elapsed_seconds,
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_convergence_table(
    option: OptionContract,
    option_type: OptionType,
    *,
    path_grid: list[int] | tuple[int, ...] = (500, 1_000, 2_500, 5_000, 10_000),
    steps: int = 252,
    seed: int = 7,
    antithetic: bool = True,
    control_variate: bool = True,
) -> pd.DataFrame:
    benchmark = black_scholes_price(option, option_type)
    rows: list[dict[str, float]] = []
    for paths in path_grid:
        result = monte_carlo_price(
            option,
            option_type,
            paths=paths,
            steps=steps,
            seed=seed,
            antithetic=antithetic,
            control_variate=control_variate,
        )
        rows.append(
            {
                "paths": float(paths),
                "price": result.price,
                "std_error": result.std_error,
                "ci_width": result.confidence_interval[1] - result.confidence_interval[0],
                "abs_error": abs(result.price - benchmark),
                "runtime_seconds": result.elapsed_seconds,
            }
        )
    return pd.DataFrame(rows)
