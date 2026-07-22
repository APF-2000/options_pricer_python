"""Monte Carlo pricing with path samples for visual inspection."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random

from options_pricer.contracts import EuropeanOption


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    discounted_payoffs: list[float]
    terminal_prices: list[float]
    sample_paths: list[list[float]]


def _simulate_path(
    option: EuropeanOption,
    *,
    steps: int,
    shocks: list[float],
) -> list[float]:
    dt = option.time_to_maturity / steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * sqrt(dt)
    spot = option.spot
    path = [spot]

    for shock in shocks:
        spot *= exp(drift + diffusion * shock)
        path.append(spot)
    return path


def monte_carlo_price(
    option: EuropeanOption,
    option_type: str,
    *,
    simulations: int = 10_000,
    steps: int = 252,
    seed: int | None = 7,
    antithetic: bool = True,
    path_samples: int = 8,
) -> MonteCarloResult:
    if simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    tau = option.time_to_maturity
    if tau == 0.0:
        payoff = option.intrinsic_value(option_type)
        return MonteCarloResult(
            price=payoff,
            std_error=0.0,
            confidence_interval=(payoff, payoff),
            discounted_payoffs=[payoff],
            terminal_prices=[option.spot],
            sample_paths=[[option.spot]],
        )

    if option.volatility == 0.0:
        terminal_spot = option.spot * exp(
            (option.rate - option.dividend_yield) * option.time_to_maturity
        )
        payoff = exp(-option.rate * tau) * option.intrinsic_value(
            option_type,
            spot=terminal_spot,
        )
        deterministic_path = [
            option.spot + (terminal_spot - option.spot) * index / steps
            for index in range(steps + 1)
        ]
        return MonteCarloResult(
            price=payoff,
            std_error=0.0,
            confidence_interval=(payoff, payoff),
            discounted_payoffs=[payoff],
            terminal_prices=[terminal_spot],
            sample_paths=[deterministic_path],
        )

    rng = Random(seed)
    discounted_payoffs: list[float] = []
    terminal_prices: list[float] = []
    sample_paths: list[list[float]] = []

    simulation_index = 0
    while simulation_index < simulations:
        shocks = [rng.gauss(0.0, 1.0) for _ in range(steps)]
        path = _simulate_path(option, steps=steps, shocks=shocks)
        terminal_spot = path[-1]
        payoff = exp(-option.rate * tau) * option.intrinsic_value(
            option_type,
            spot=terminal_spot,
        )
        discounted_payoffs.append(payoff)
        terminal_prices.append(terminal_spot)
        if len(sample_paths) < path_samples:
            sample_paths.append(path)
        simulation_index += 1

        if antithetic and simulation_index < simulations:
            antithetic_shocks = [-shock for shock in shocks]
            antithetic_path = _simulate_path(
                option,
                steps=steps,
                shocks=antithetic_shocks,
            )
            antithetic_terminal = antithetic_path[-1]
            antithetic_payoff = exp(-option.rate * tau) * option.intrinsic_value(
                option_type,
                spot=antithetic_terminal,
            )
            discounted_payoffs.append(antithetic_payoff)
            terminal_prices.append(antithetic_terminal)
            if len(sample_paths) < path_samples:
                sample_paths.append(antithetic_path)
            simulation_index += 1

    count = len(discounted_payoffs)
    mean = sum(discounted_payoffs) / count
    if count == 1:
        variance = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in discounted_payoffs) / (count - 1)
    std_error = sqrt(variance / count)
    interval = (mean - 1.96 * std_error, mean + 1.96 * std_error)

    return MonteCarloResult(
        price=mean,
        std_error=std_error,
        confidence_interval=interval,
        discounted_payoffs=discounted_payoffs,
        terminal_prices=terminal_prices,
        sample_paths=sample_paths,
    )
