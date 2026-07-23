import pytest

from options_pricer import EuropeanOption, asian_option_price, black_scholes_price


def test_asian_call_price_is_below_european_call_price_for_same_contract() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    european_price = black_scholes_price(option, "call")
    asian_price = asian_option_price(option, "call", paths=20_000, steps=100, seed=11)
    assert asian_price.price < european_price
    assert asian_price.confidence_interval[0] <= asian_price.price <= asian_price.confidence_interval[1]
