"""Advanced extensions built on the core pricing stack."""

from __future__ import annotations

import numpy as np

from .instruments import OptionContract, OptionType
from .monte_carlo import MonteCarloResult, simulate_gbm_paths
from .validation import ensure_option_type


def asian_option_price(
    option: OptionContract,
    option_type: OptionType,
    *,
    paths: int = 10_000,
    steps: int = 252,
    seed: int | None = 7,
    antithetic: bool = True,
    sample_path_count: int = 8,
    include_initial_spot: bool = True,
) -> MonteCarloResult:
    """Price an arithmetic-average Asian option using Monte Carlo."""

    ensure_option_type(option_type)
    paths_matrix = simulate_gbm_paths(
        option,
        paths=paths,
        steps=steps,
        seed=seed,
        antithetic=antithetic,
    )
    averaging_slice = paths_matrix if include_initial_spot else paths_matrix[:, 1:]
    average_spots = averaging_slice.mean(axis=1)
    if option_type == "call":
        payoff = np.maximum(average_spots - option.strike, 0.0)
    else:
        payoff = np.maximum(option.strike - average_spots, 0.0)
    discounted_payoffs = np.exp(-option.rate * option.time_to_maturity) * payoff
    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs)))
    confidence_interval = (price - 1.96 * std_error, price + 1.96 * std_error)
    return MonteCarloResult(
        price=price,
        std_error=std_error,
        confidence_interval=confidence_interval,
        discounted_payoffs=discounted_payoffs,
        terminal_prices=paths_matrix[:, -1],
        sample_paths=paths_matrix[:sample_path_count],
        elapsed_seconds=0.0,
        method="asian_arithmetic_monte_carlo",
        control_variate_beta=None,
    )
