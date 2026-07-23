"""Validation helpers and finite-difference diagnostics."""

from __future__ import annotations

from typing import Callable

from .instruments import OptionContract, OptionGreeks, OptionType


PriceFunction = Callable[[OptionContract, OptionType], float]


def ensure_option_type(option_type: OptionType) -> OptionType:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def put_call_parity_rhs(option: OptionContract) -> float:
    from math import exp

    tau = option.time_to_maturity
    return option.spot * exp(-option.dividend_yield * tau) - option.strike * exp(
        -option.rate * tau
    )


def put_call_parity_gap(option: OptionContract, call_price: float, put_price: float) -> float:
    return call_price - put_price - put_call_parity_rhs(option)


def european_price_bounds(option: OptionContract, option_type: OptionType) -> tuple[float, float]:
    from math import exp

    ensure_option_type(option_type)
    tau = option.time_to_maturity
    discounted_spot = option.spot * exp(-option.dividend_yield * tau)
    discounted_strike = option.strike * exp(-option.rate * tau)
    if option_type == "call":
        lower = max(0.0, discounted_spot - discounted_strike)
        upper = discounted_spot
    else:
        lower = max(0.0, discounted_strike - discounted_spot)
        upper = discounted_strike
    return lower, upper


def validate_market_price(
    option: OptionContract,
    option_type: OptionType,
    market_price: float,
) -> None:
    lower, upper = european_price_bounds(option, option_type)
    if market_price < lower - 1e-12 or market_price > upper + 1e-12:
        raise ValueError(
            f"market price {market_price:.6f} violates no-arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}]"
        )


def finite_difference_greeks(
    option: OptionContract,
    option_type: OptionType,
    pricing_function: PriceFunction,
    *,
    spot_bump: float | None = None,
    vol_bump: float = 1e-4,
    rate_bump: float = 1e-4,
    time_bump: float = 1.0 / 365.0,
) -> OptionGreeks:
    ensure_option_type(option_type)
    spot_step = spot_bump or max(1e-4, option.spot * 1e-4)

    up_spot = option.with_spot(option.spot + spot_step)
    down_spot = option.with_spot(max(option.spot - spot_step, 1e-8))
    price_up = pricing_function(up_spot, option_type)
    price_mid = pricing_function(option, option_type)
    price_down = pricing_function(down_spot, option_type)

    delta = (price_up - price_down) / (2.0 * spot_step)
    gamma = (price_up - 2.0 * price_mid + price_down) / (spot_step**2)

    up_vol = option.with_volatility(option.volatility + vol_bump)
    down_vol = option.with_volatility(max(option.volatility - vol_bump, 1e-8))
    vega = (
        pricing_function(up_vol, option_type) - pricing_function(down_vol, option_type)
    ) / (2.0 * vol_bump)

    up_rate = OptionContract(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate + rate_bump,
        volatility=option.volatility,
        current_time=option.current_time,
        dividend_yield=option.dividend_yield,
        exercise_style=option.exercise_style,
    )
    down_rate = OptionContract(
        spot=option.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=option.rate - rate_bump,
        volatility=option.volatility,
        current_time=option.current_time,
        dividend_yield=option.dividend_yield,
        exercise_style=option.exercise_style,
    )
    rho = (
        pricing_function(up_rate, option_type) - pricing_function(down_rate, option_type)
    ) / (2.0 * rate_bump)

    if option.time_to_maturity <= time_bump:
        theta = 0.0
    else:
        advanced = option.with_current_time(option.current_time + time_bump)
        theta = (pricing_function(advanced, option_type) - price_mid) / time_bump

    return OptionGreeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
