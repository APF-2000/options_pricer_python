import pytest

from options_pricer import (
    AmericanOption,
    EuropeanOption,
    binomial_convergence_table,
    black_scholes_price,
    cox_ross_rubinstein_price,
)


def test_binomial_converges_to_black_scholes_for_european_call() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    result = cox_ross_rubinstein_price(option, "call", steps=400)
    benchmark = black_scholes_price(option, "call")
    assert result.price == pytest.approx(benchmark, abs=0.02)


def test_american_put_is_at_least_as_valuable_as_european_put() -> None:
    european = EuropeanOption(
        spot=80.0,
        strike=100.0,
        maturity=1.5,
        rate=0.08,
        volatility=0.30,
    )
    american = AmericanOption(
        spot=european.spot,
        strike=european.strike,
        maturity=european.maturity,
        current_time=european.current_time,
        rate=european.rate,
        volatility=european.volatility,
        dividend_yield=european.dividend_yield,
    )
    euro_price = cox_ross_rubinstein_price(european, "put", steps=250).price
    american_result = cox_ross_rubinstein_price(american, "put", steps=250, return_tree=True)
    assert american_result.price >= euro_price
    assert american_result.early_exercise_count > 0


def test_american_call_without_dividends_matches_european_value() -> None:
    european = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.2,
    )
    american = AmericanOption(
        spot=european.spot,
        strike=european.strike,
        maturity=european.maturity,
        current_time=european.current_time,
        rate=european.rate,
        volatility=european.volatility,
        dividend_yield=european.dividend_yield,
    )
    euro_price = cox_ross_rubinstein_price(european, "call", steps=300).price
    amer_price = cox_ross_rubinstein_price(american, "call", steps=300).price
    assert amer_price == pytest.approx(euro_price, abs=0.02)


def test_return_tree_enables_tree_rendering_data() -> None:
    option = AmericanOption(
        spot=80.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.3,
    )
    result = cox_ross_rubinstein_price(option, "put", steps=4, return_tree=True)
    assert result.stock_tree is not None
    assert result.option_tree is not None
    assert result.early_exercise_tree is not None
    assert len(result.stock_tree) == 5


def test_convergence_table_reports_decreasing_error() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    table = binomial_convergence_table(option, "call", step_grid=(10, 50, 200))
    assert list(table.columns) == [
        "steps",
        "binomial_price",
        "black_scholes_price",
        "abs_error",
        "runtime_seconds",
    ]
    assert table["abs_error"].iloc[-1] < table["abs_error"].iloc[0]
