"""Implied-volatility inversion utilities."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from scipy.optimize import brentq

from .black_scholes import black_scholes_greeks, black_scholes_price
from .instruments import OptionContract, OptionType
from .validation import ensure_option_type, validate_market_price


class ImpliedVolatilityError(ValueError):
    """Raised when implied volatility cannot be recovered robustly."""


def implied_volatility(
    option: OptionContract,
    option_type: OptionType,
    market_price: float,
    *,
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    lower_bound: float = 1e-8,
    upper_bound: float = 5.0,
) -> float:
    ensure_option_type(option_type)
    validate_market_price(option, option_type, market_price)

    if option.time_to_maturity == 0.0:
        intrinsic = option.intrinsic_value(option_type)
        if abs(market_price - intrinsic) <= tolerance:
            return 0.0
        raise ImpliedVolatilityError(
            "cannot infer implied volatility after expiry when market price differs from intrinsic value"
        )

    sigma = max(initial_guess, lower_bound)
    for _ in range(max_iterations):
        trial = option.with_volatility(sigma)
        price = black_scholes_price(trial, option_type)
        diff = price - market_price
        if abs(diff) <= tolerance:
            return sigma
        vega = black_scholes_greeks(trial, option_type).vega
        if abs(vega) < 1e-10:
            break
        sigma_next = sigma - diff / vega
        if sigma_next <= lower_bound or sigma_next >= upper_bound:
            break
        sigma = sigma_next

    def objective(volatility: float) -> float:
        return black_scholes_price(option.with_volatility(volatility), option_type) - market_price

    try:
        return float(brentq(objective, lower_bound, upper_bound, xtol=tolerance, maxiter=max_iterations))
    except ValueError as exc:
        raise ImpliedVolatilityError(
            "failed to bracket an implied-volatility root within the configured bounds"
        ) from exc


def implied_volatility_frame(
    option_chain: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    price_column: str = "price",
    strike_column: str = "strike",
    maturity_column: str = "maturity",
    option_type_column: str = "option_type",
) -> pd.DataFrame:
    rows = []
    for record in option_chain.to_dict(orient="records"):
        option = OptionContract(
            spot=spot,
            strike=float(record[strike_column]),
            maturity=float(record[maturity_column]),
            current_time=0.0,
            rate=rate,
            volatility=0.2,
            dividend_yield=dividend_yield,
        )
        market_price = float(record[price_column])
        option_type = str(record[option_type_column])
        vol = implied_volatility(option, option_type, market_price)
        rows.append({**record, "implied_volatility": vol})
    return pd.DataFrame(rows)


def build_volatility_surface(
    option_chain: pd.DataFrame,
    *,
    strike_column: str = "strike",
    maturity_column: str = "maturity",
    implied_volatility_column: str = "implied_volatility",
) -> pd.DataFrame:
    return option_chain.pivot(
        index=maturity_column,
        columns=strike_column,
        values=implied_volatility_column,
    ).sort_index()
