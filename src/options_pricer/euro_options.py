"""Convenience helpers for constructing European options."""

from __future__ import annotations

from options_pricer.contracts import EuropeanOption


def make_european_option(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    current_time: float = 0.0,
    dividend_yield: float = 0.0,
) -> EuropeanOption:
    return EuropeanOption(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        current_time=current_time,
        dividend_yield=dividend_yield,
    )
