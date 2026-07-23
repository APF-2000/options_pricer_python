import pandas as pd
import pytest

from options_pricer import (
    EuropeanOption,
    build_volatility_surface,
    implied_volatility,
    implied_volatility_frame,
)
from options_pricer.black_scholes import black_scholes_price


def test_implied_volatility_round_trip_recovers_input_sigma() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=105.0,
        maturity=1.25,
        rate=0.03,
        volatility=0.24,
        dividend_yield=0.01,
    )
    market_price = black_scholes_price(option, "call")
    recovered = implied_volatility(option, "call", market_price)
    assert recovered == pytest.approx(option.volatility, abs=1e-8)


def test_implied_volatility_rejects_arbitrage_violating_prices() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(option, "call", market_price=150.0)


def test_implied_volatility_frame_and_surface_build_cleanly() -> None:
    spot = 100.0
    rows = []
    for maturity, sigma in ((0.5, 0.20), (1.0, 0.22)):
        for strike in (90.0, 100.0, 110.0):
            option = EuropeanOption(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=0.03,
                volatility=sigma,
            )
            rows.append(
                {
                    "strike": strike,
                    "maturity": maturity,
                    "option_type": "call",
                    "price": black_scholes_price(option, "call"),
                }
            )
    chain = pd.DataFrame(rows)
    with_iv = implied_volatility_frame(chain, spot=spot, rate=0.03)
    surface = build_volatility_surface(with_iv)
    assert "implied_volatility" in with_iv.columns
    assert surface.shape == (2, 3)
