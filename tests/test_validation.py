import math

import pytest

from options_pricer import (
    EuropeanOption,
    black_scholes_price,
    european_price_bounds,
    finite_difference_greeks,
    put_call_parity_rhs,
    validate_market_price,
)


def test_no_arbitrage_bounds_are_respected_by_black_scholes_prices() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=95.0,
        maturity=1.0,
        rate=0.04,
        volatility=0.2,
        dividend_yield=0.01,
    )
    for option_type in ("call", "put"):
        lower, upper = european_price_bounds(option, option_type)
        price = black_scholes_price(option, option_type)
        assert lower <= price <= upper


def test_market_price_validation_flags_invalid_price() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=95.0,
        maturity=1.0,
        rate=0.04,
        volatility=0.2,
    )
    with pytest.raises(ValueError, match="no-arbitrage"):
        validate_market_price(option, "put", 200.0)


def test_put_call_parity_rhs_matches_known_formula() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
    )
    rhs = put_call_parity_rhs(option)
    expected = 100.0 * math.exp(-0.02) - 100.0 * math.exp(-0.05)
    assert rhs == pytest.approx(expected, abs=1e-12)


def test_finite_difference_greeks_returns_reasonable_values() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    greeks = finite_difference_greeks(option, "call", black_scholes_price)
    assert 0.0 < greeks.delta < 1.0
    assert greeks.gamma > 0.0
    assert greeks.vega > 0.0
