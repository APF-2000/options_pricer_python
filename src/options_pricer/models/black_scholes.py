"""Black-Scholes pricing and greeks for European options."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, inf, log, pi, sqrt

from options_pricer.contracts import EuropeanOption


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _deterministic_terminal_spot(option: EuropeanOption) -> float:
    return option.spot * exp(
        (option.rate - option.dividend_yield) * option.time_to_maturity
    )


def _deterministic_price(option: EuropeanOption, option_type: str) -> float:
    terminal_spot = _deterministic_terminal_spot(option)
    payoff = option.intrinsic_value(option_type, spot=terminal_spot)
    return exp(-option.rate * option.time_to_maturity) * payoff


def d1(option: EuropeanOption) -> float:
    tau = option.time_to_maturity
    sigma = option.volatility
    if tau == 0.0 or sigma == 0.0:
        terminal_spot = _deterministic_terminal_spot(option)
        if terminal_spot > option.strike:
            return inf
        if terminal_spot < option.strike:
            return -inf
        return 0.0
    numerator = log(option.spot / option.strike) + (
        option.rate - option.dividend_yield + 0.5 * sigma * sigma
    ) * tau
    return numerator / (sigma * sqrt(tau))


def d2(option: EuropeanOption) -> float:
    tau = option.time_to_maturity
    sigma = option.volatility
    if tau == 0.0 or sigma == 0.0:
        return d1(option)
    return d1(option) - sigma * sqrt(tau)


def black_scholes_price(option: EuropeanOption, option_type: str) -> float:
    tau = option.time_to_maturity
    if tau == 0.0:
        return option.intrinsic_value(option_type)
    if option.volatility == 0.0:
        return _deterministic_price(option, option_type)

    d_1 = d1(option)
    d_2 = d2(option)
    discounted_spot = option.spot * exp(-option.dividend_yield * tau)
    discounted_strike = option.strike * exp(-option.rate * tau)

    if option_type == "call":
        return discounted_spot * normal_cdf(d_1) - discounted_strike * normal_cdf(d_2)
    if option_type == "put":
        return discounted_strike * normal_cdf(-d_2) - discounted_spot * normal_cdf(-d_1)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(option: EuropeanOption, option_type: str) -> OptionGreeks:
    tau = option.time_to_maturity
    if tau == 0.0 or option.volatility == 0.0:
        intrinsic_delta = 0.0
        if option_type == "call" and option.spot > option.strike:
            intrinsic_delta = 1.0
        if option_type == "put" and option.spot < option.strike:
            intrinsic_delta = -1.0
        return OptionGreeks(
            delta=intrinsic_delta,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
        )

    d_1 = d1(option)
    d_2 = d2(option)
    sigma_root_tau = option.volatility * sqrt(tau)
    discounted_spot = option.spot * exp(-option.dividend_yield * tau)
    discounted_strike = option.strike * exp(-option.rate * tau)
    pdf = normal_pdf(d_1)

    gamma = exp(-option.dividend_yield * tau) * pdf / (option.spot * sigma_root_tau)
    vega = discounted_spot * pdf * sqrt(tau)

    if option_type == "call":
        delta = exp(-option.dividend_yield * tau) * normal_cdf(d_1)
        theta = (
            -(discounted_spot * pdf * option.volatility) / (2.0 * sqrt(tau))
            - option.rate * discounted_strike * normal_cdf(d_2)
            + option.dividend_yield * discounted_spot * normal_cdf(d_1)
        )
        rho = tau * discounted_strike * normal_cdf(d_2)
    elif option_type == "put":
        delta = exp(-option.dividend_yield * tau) * (normal_cdf(d_1) - 1.0)
        theta = (
            -(discounted_spot * pdf * option.volatility) / (2.0 * sqrt(tau))
            + option.rate * discounted_strike * normal_cdf(-d_2)
            - option.dividend_yield * discounted_spot * normal_cdf(-d_1)
        )
        rho = -tau * discounted_strike * normal_cdf(-d_2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return OptionGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )


def european_call_option(
    sigma: float,
    T: float,
    t: float,
    S: float,
    K: float,
    r: float,
    q: float = 0.0,
) -> float:
    option = EuropeanOption(
        spot=S,
        strike=K,
        maturity=T,
        rate=r,
        volatility=sigma,
        current_time=t,
        dividend_yield=q,
    )
    return black_scholes_price(option, "call")


def european_put_option(
    sigma: float,
    T: float,
    t: float,
    S: float,
    K: float,
    r: float,
    q: float = 0.0,
) -> float:
    option = EuropeanOption(
        spot=S,
        strike=K,
        maturity=T,
        rate=r,
        volatility=sigma,
        current_time=t,
        dividend_yield=q,
    )
    return black_scholes_price(option, "put")
