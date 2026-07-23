"""Discrete delta-hedging simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import black_scholes_greeks_inputs, black_scholes_price
from .instruments import OptionContract, OptionType
from .monte_carlo import simulate_gbm_paths
from .validation import ensure_option_type


@dataclass(frozen=True)
class HedgingSimulationResult:
    """Delta-hedging simulation output."""

    hedging_errors: np.ndarray
    mean_error: float
    std_error: float
    quantiles: dict[str, float]
    sample_paths: np.ndarray
    sample_cash_paths: np.ndarray
    sample_delta_paths: np.ndarray


def simulate_delta_hedge(
    option: OptionContract,
    option_type: OptionType,
    *,
    paths: int = 1_000,
    steps: int = 252,
    rebalance_every: int = 1,
    seed: int | None = 7,
    realised_volatility: float | None = None,
    implied_volatility: float | None = None,
    transaction_cost_bps: float = 0.0,
    drift: float | None = None,
    sample_path_count: int = 8,
) -> HedgingSimulationResult:
    ensure_option_type(option_type)
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be a positive integer")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    hedge_volatility = option.volatility if implied_volatility is None else implied_volatility
    realised_option = option.with_volatility(
        option.volatility if realised_volatility is None else realised_volatility
    )
    hedge_option = option.with_volatility(hedge_volatility)
    paths_matrix = simulate_gbm_paths(
        realised_option,
        paths=paths,
        steps=steps,
        seed=seed,
        antithetic=False,
        drift=drift,
    )

    dt = option.time_to_maturity / steps
    discount = np.exp(option.rate * dt)
    transaction_cost_rate = transaction_cost_bps / 10_000.0

    initial_price = black_scholes_price(hedge_option, option_type)
    initial_delta = float(
        black_scholes_greeks_inputs(
            option.spot,
            option.strike,
            option.time_to_maturity,
            option.rate,
            hedge_volatility,
            option_type,
            option.dividend_yield,
        )["delta"]
    )
    cash = np.full(paths, initial_price - initial_delta * option.spot, dtype=float)
    cash -= transaction_cost_rate * abs(initial_delta) * option.spot
    delta = np.full(paths, initial_delta, dtype=float)

    sample_cash_paths = [cash[:sample_path_count].copy()]
    sample_delta_paths = [delta[:sample_path_count].copy()]

    for step in range(1, steps + 1):
        cash *= discount
        if step < steps and step % rebalance_every == 0:
            tau_remaining = option.time_to_maturity - step * dt
            deltas = np.asarray(
                black_scholes_greeks_inputs(
                    paths_matrix[:, step],
                    option.strike,
                    tau_remaining,
                    option.rate,
                    hedge_volatility,
                    option_type,
                    option.dividend_yield,
                )["delta"],
                dtype=float,
            )
            trade = deltas - delta
            costs = transaction_cost_rate * np.abs(trade) * paths_matrix[:, step]
            cash -= trade * paths_matrix[:, step] + costs
            delta = deltas
        if step <= steps:
            sample_cash_paths.append(cash[:sample_path_count].copy())
            sample_delta_paths.append(delta[:sample_path_count].copy())

    terminal_spots = paths_matrix[:, -1]
    if option_type == "call":
        payoff = np.maximum(terminal_spots - option.strike, 0.0)
    else:
        payoff = np.maximum(option.strike - terminal_spots, 0.0)
    hedging_errors = cash + delta * terminal_spots - payoff
    quantile_values = np.quantile(hedging_errors, [0.05, 0.5, 0.95])

    return HedgingSimulationResult(
        hedging_errors=hedging_errors,
        mean_error=float(hedging_errors.mean()),
        std_error=float(hedging_errors.std(ddof=1)),
        quantiles={
            "q05": float(quantile_values[0]),
            "q50": float(quantile_values[1]),
            "q95": float(quantile_values[2]),
        },
        sample_paths=paths_matrix[:sample_path_count],
        sample_cash_paths=np.asarray(sample_cash_paths).T,
        sample_delta_paths=np.asarray(sample_delta_paths).T,
    )
