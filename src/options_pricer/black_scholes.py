"""Black-Scholes pricing, vectorised utilities, and analytical greeks."""

from __future__ import annotations

from dataclasses import asdict
from math import exp
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .instruments import OptionContract, OptionGreeks, OptionType
from .validation import ensure_option_type


def _to_numpy(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


def _maybe_scalar(value: NDArray[np.float64] | np.float64) -> float | NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


def d1_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    s = _to_numpy(spot)
    k = _to_numpy(strike)
    t = _to_numpy(tau)
    r = _to_numpy(rate)
    sigma = _to_numpy(volatility)
    q = _to_numpy(dividend_yield)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.log(s / k) + (r - q + 0.5 * sigma**2) * t
        denominator = sigma * np.sqrt(t)
        values = numerator / denominator

    deterministic_forward = s * np.exp((r - q) * t)
    values = np.where((t == 0.0) | (sigma == 0.0), np.sign(deterministic_forward - k) * np.inf, values)
    values = np.where(
        ((t == 0.0) | (sigma == 0.0)) & (np.isclose(deterministic_forward, k)),
        0.0,
        values,
    )
    return _maybe_scalar(values)


def d2_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    d1 = _to_numpy(d1_inputs(spot, strike, tau, rate, volatility, dividend_yield))
    sigma = _to_numpy(volatility)
    t = _to_numpy(tau)
    values = d1 - sigma * np.sqrt(t)
    values = np.where((t == 0.0) | (sigma == 0.0), d1, values)
    return _maybe_scalar(values)


def black_scholes_price_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> float | NDArray[np.float64]:
    ensure_option_type(option_type)
    s = _to_numpy(spot)
    k = _to_numpy(strike)
    t = _to_numpy(tau)
    r = _to_numpy(rate)
    sigma = _to_numpy(volatility)
    q = _to_numpy(dividend_yield)

    if np.any(s <= 0):
        raise ValueError("spot must be strictly positive")
    if np.any(k <= 0):
        raise ValueError("strike must be strictly positive")
    if np.any(t < 0):
        raise ValueError("time to maturity must be non-negative")
    if np.any(sigma < 0):
        raise ValueError("volatility must be non-negative")

    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    intrinsic = np.maximum(s - k, 0.0) if option_type == "call" else np.maximum(k - s, 0.0)

    deterministic_forward = s * np.exp((r - q) * t)
    deterministic_payoff = np.maximum(deterministic_forward - k, 0.0)
    if option_type == "put":
        deterministic_payoff = np.maximum(k - deterministic_forward, 0.0)
    deterministic_price = np.exp(-r * t) * deterministic_payoff

    d1 = _to_numpy(d1_inputs(s, k, t, r, sigma, q))
    d2 = _to_numpy(d2_inputs(s, k, t, r, sigma, q))
    if option_type == "call":
        values = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        values = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)

    values = np.where(t == 0.0, intrinsic, values)
    values = np.where((t > 0.0) & (sigma == 0.0), deterministic_price, values)
    return _maybe_scalar(values)


def black_scholes_greeks_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> dict[str, float | NDArray[np.float64]]:
    ensure_option_type(option_type)
    s = _to_numpy(spot)
    k = _to_numpy(strike)
    t = _to_numpy(tau)
    r = _to_numpy(rate)
    sigma = _to_numpy(volatility)
    q = _to_numpy(dividend_yield)

    d1 = _to_numpy(d1_inputs(s, k, t, r, sigma, q))
    d2 = _to_numpy(d2_inputs(s, k, t, r, sigma, q))
    pdf = norm.pdf(d1)
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    sigma_root_tau = sigma * np.sqrt(t)

    gamma = np.where(
        (t == 0.0) | (sigma == 0.0),
        0.0,
        np.exp(-q * t) * pdf / (s * sigma_root_tau),
    )
    vega = np.where((t == 0.0) | (sigma == 0.0), 0.0, discounted_spot * pdf * np.sqrt(t))

    if option_type == "call":
        delta = np.where(
            (t == 0.0) | (sigma == 0.0),
            np.where(s > k, 1.0, 0.0),
            np.exp(-q * t) * norm.cdf(d1),
        )
        theta = np.where(
            (t == 0.0) | (sigma == 0.0),
            0.0,
            -(discounted_spot * pdf * sigma) / (2.0 * np.sqrt(t))
            - r * discounted_strike * norm.cdf(d2)
            + q * discounted_spot * norm.cdf(d1),
        )
        rho = np.where(
            (t == 0.0) | (sigma == 0.0),
            0.0,
            t * discounted_strike * norm.cdf(d2),
        )
    else:
        delta = np.where(
            (t == 0.0) | (sigma == 0.0),
            np.where(s < k, -1.0, 0.0),
            np.exp(-q * t) * (norm.cdf(d1) - 1.0),
        )
        theta = np.where(
            (t == 0.0) | (sigma == 0.0),
            0.0,
            -(discounted_spot * pdf * sigma) / (2.0 * np.sqrt(t))
            + r * discounted_strike * norm.cdf(-d2)
            - q * discounted_spot * norm.cdf(-d1),
        )
        rho = np.where(
            (t == 0.0) | (sigma == 0.0),
            0.0,
            -t * discounted_strike * norm.cdf(-d2),
        )

    return {
        "delta": _maybe_scalar(delta),
        "gamma": _maybe_scalar(gamma),
        "vega": _maybe_scalar(vega),
        "theta": _maybe_scalar(theta),
        "rho": _maybe_scalar(rho),
    }


def d1(option: OptionContract) -> float:
    return float(
        d1_inputs(
            option.spot,
            option.strike,
            option.time_to_maturity,
            option.rate,
            option.volatility,
            option.dividend_yield,
        )
    )


def d2(option: OptionContract) -> float:
    return float(
        d2_inputs(
            option.spot,
            option.strike,
            option.time_to_maturity,
            option.rate,
            option.volatility,
            option.dividend_yield,
        )
    )


def black_scholes_price(option: OptionContract, option_type: OptionType) -> float:
    return float(
        black_scholes_price_inputs(
            option.spot,
            option.strike,
            option.time_to_maturity,
            option.rate,
            option.volatility,
            option_type,
            option.dividend_yield,
        )
    )


def black_scholes_greeks(option: OptionContract, option_type: OptionType) -> OptionGreeks:
    values = black_scholes_greeks_inputs(
        option.spot,
        option.strike,
        option.time_to_maturity,
        option.rate,
        option.volatility,
        option_type,
        option.dividend_yield,
    )
    return OptionGreeks(**{name: float(value) for name, value in values.items()})


def european_call_option(
    sigma: float,
    T: float,
    t: float,
    S: float,
    K: float,
    r: float,
    q: float = 0.0,
) -> float:
    option = OptionContract(
        spot=S,
        strike=K,
        maturity=T,
        rate=r,
        volatility=sigma,
        current_time=t,
        dividend_yield=q,
        exercise_style="european",
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
    option = OptionContract(
        spot=S,
        strike=K,
        maturity=T,
        rate=r,
        volatility=sigma,
        current_time=t,
        dividend_yield=q,
        exercise_style="european",
    )
    return black_scholes_price(option, "put")
