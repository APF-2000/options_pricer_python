import math

import numpy as np
import pytest

from options_pricer import (
    EuropeanOption,
    black_scholes_greeks,
    black_scholes_price,
    black_scholes_price_inputs,
    european_call_option,
    european_put_option,
    finite_difference_greeks,
    put_call_parity_gap,
)


def test_black_scholes_matches_reference_values() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    assert european_call_option(0.2, 1.0, 0.0, 100.0, 100.0, 0.05) == pytest.approx(10.4506, abs=1e-4)
    assert european_put_option(0.2, 1.0, 0.0, 100.0, 100.0, 0.05) == pytest.approx(5.5735, abs=1e-4)
    assert black_scholes_price(option, "call") == pytest.approx(10.450583572185565, abs=1e-10)


def test_put_call_parity_holds_with_dividend_yield() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=95.0,
        maturity=1.0,
        rate=0.04,
        volatility=0.25,
        dividend_yield=0.01,
    )
    call_price = black_scholes_price(option, "call")
    put_price = black_scholes_price(option, "put")
    assert put_call_parity_gap(option, call_price, put_price) == pytest.approx(0.0, abs=1e-10)


def test_expiry_and_zero_volatility_behaviour_are_sensible() -> None:
    expired = EuropeanOption(
        spot=120.0,
        strike=100.0,
        maturity=1.0,
        current_time=1.0,
        rate=0.05,
        volatility=0.2,
    )
    assert black_scholes_price(expired, "call") == 20.0
    assert black_scholes_price(expired, "put") == 0.0

    deterministic = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.0,
    )
    expected = max(100.0 * math.exp(0.05) - 100.0, 0.0) * math.exp(-0.05)
    assert black_scholes_price(deterministic, "call") == pytest.approx(expected, abs=1e-8)


def test_vectorised_inputs_return_array_prices() -> None:
    prices = black_scholes_price_inputs(
        spot=np.array([90.0, 100.0, 110.0]),
        strike=100.0,
        tau=1.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.0,
        option_type="call",
    )
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) > 0.0)


def test_analytical_greeks_match_finite_difference() -> None:
    option = EuropeanOption(
        spot=103.0,
        strike=100.0,
        maturity=1.3,
        rate=0.03,
        volatility=0.22,
        dividend_yield=0.01,
    )
    analytical = black_scholes_greeks(option, "call")
    numerical = finite_difference_greeks(option, "call", black_scholes_price)
    assert analytical.delta == pytest.approx(numerical.delta, abs=1e-4)
    assert analytical.gamma == pytest.approx(numerical.gamma, abs=1e-5)
    assert analytical.vega == pytest.approx(numerical.vega, rel=5e-4)
    assert analytical.theta == pytest.approx(numerical.theta, rel=1e-3, abs=2e-3)
    assert analytical.rho == pytest.approx(numerical.rho, rel=5e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot": -1.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2}, "spot"),
        ({"spot": 100.0, "strike": 0.0, "maturity": 1.0, "rate": 0.05, "volatility": 0.2}, "strike"),
        ({"spot": 100.0, "strike": 100.0, "maturity": 0.0, "current_time": 1.0, "rate": 0.05, "volatility": 0.2}, "maturity"),
        ({"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "volatility": -0.1}, "volatility"),
    ],
)
def test_invalid_contract_inputs_raise_value_errors(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EuropeanOption(**kwargs)
