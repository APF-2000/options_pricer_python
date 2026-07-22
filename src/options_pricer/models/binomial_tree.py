"""Cox-Ross-Rubinstein binomial tree for European options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from options_pricer.contracts import EuropeanOption


@dataclass(frozen=True)
class BinomialTreeResult:
    price: float
    up_factor: float
    down_factor: float
    risk_neutral_probability: float
    stock_tree: list[list[float]]
    option_tree: list[list[float]]


def cox_ross_rubinstein_price(
    option: EuropeanOption,
    option_type: str,
    *,
    steps: int = 100,
) -> BinomialTreeResult:
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    tau = option.time_to_maturity
    if tau == 0.0:
        payoff = option.intrinsic_value(option_type)
        return BinomialTreeResult(
            price=payoff,
            up_factor=1.0,
            down_factor=1.0,
            risk_neutral_probability=0.5,
            stock_tree=[[option.spot]],
            option_tree=[[payoff]],
        )

    if option.volatility == 0.0:
        terminal_spot = option.spot * exp(
            (option.rate - option.dividend_yield) * option.time_to_maturity
        )
        discounted_payoff = exp(-option.rate * option.time_to_maturity) * option.intrinsic_value(
            option_type,
            spot=terminal_spot,
        )
        return BinomialTreeResult(
            price=discounted_payoff,
            up_factor=1.0,
            down_factor=1.0,
            risk_neutral_probability=1.0,
            stock_tree=[[option.spot], [terminal_spot]],
            option_tree=[[discounted_payoff], [option.intrinsic_value(option_type, spot=terminal_spot)]],
        )

    dt = tau / steps
    growth = exp((option.rate - option.dividend_yield) * dt)
    up_factor = exp(option.volatility * sqrt(dt))
    down_factor = 1.0 / up_factor
    probability = (growth - down_factor) / (up_factor - down_factor)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "risk-neutral probability is outside [0, 1]; "
            "try increasing the number of steps"
        )

    stock_tree: list[list[float]] = []
    for level in range(steps + 1):
        prices = []
        for down_moves in range(level + 1):
            up_moves = level - down_moves
            price = option.spot * (up_factor ** up_moves) * (down_factor ** down_moves)
            prices.append(price)
        stock_tree.append(prices)

    option_tree: list[list[float]] = [[] for _ in range(steps + 1)]
    option_tree[-1] = [
        option.intrinsic_value(option_type, spot=price) for price in stock_tree[-1]
    ]

    discount = exp(-option.rate * dt)
    for level in range(steps - 1, -1, -1):
        values = []
        for node in range(level + 1):
            continuation = discount * (
                probability * option_tree[level + 1][node]
                + (1.0 - probability) * option_tree[level + 1][node + 1]
            )
            values.append(continuation)
        option_tree[level] = values

    return BinomialTreeResult(
        price=option_tree[0][0],
        up_factor=up_factor,
        down_factor=down_factor,
        risk_neutral_probability=probability,
        stock_tree=stock_tree,
        option_tree=option_tree,
    )
