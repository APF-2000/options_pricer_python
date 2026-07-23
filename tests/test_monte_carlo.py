import numpy as np
import pytest

from options_pricer import (
    EuropeanOption,
    black_scholes_price,
    compare_monte_carlo_methods,
    monte_carlo_convergence_table,
    monte_carlo_price,
)


def test_monte_carlo_confidence_interval_contains_black_scholes_benchmark() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    benchmark = black_scholes_price(option, "call")
    result = monte_carlo_price(option, "call", paths=20_000, steps=100, seed=42)
    assert result.confidence_interval[0] <= benchmark <= result.confidence_interval[1]


def test_variance_reduction_improves_standard_error_or_interval_width() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    comparison = compare_monte_carlo_methods(option, "call", paths=8_000, steps=100, seed=123)
    standard = comparison.loc[comparison["method"] == "standard"].iloc[0]
    control = comparison.loc[comparison["method"] == "antithetic+control_variate"].iloc[0]
    assert control["ci_width"] < standard["ci_width"]


def test_convergence_table_ci_width_shrinks_with_more_paths() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    table = monte_carlo_convergence_table(
        option,
        "call",
        path_grid=(500, 2_000, 8_000),
        steps=80,
        seed=7,
    )
    assert table["ci_width"].iloc[-1] < table["ci_width"].iloc[0]


def test_sample_paths_and_terminal_prices_have_expected_shapes() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    )
    result = monte_carlo_price(
        option,
        "put",
        paths=256,
        steps=30,
        seed=11,
        antithetic=False,
        sample_path_count=5,
    )
    assert result.sample_paths.shape == (5, 31)
    assert result.terminal_prices.shape == (256,)
    assert result.discounted_payoffs.shape == (256,)
    assert np.all(result.discounted_payoffs >= 0.0)
