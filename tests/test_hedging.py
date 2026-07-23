import pytest

from options_pricer import EuropeanOption, simulate_delta_hedge


def test_more_frequent_rebalancing_reduces_hedging_error_dispersion() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
    )
    coarse = simulate_delta_hedge(option, "call", paths=2_000, steps=60, rebalance_every=10, seed=21)
    fine = simulate_delta_hedge(option, "call", paths=2_000, steps=60, rebalance_every=1, seed=21)
    assert fine.std_error < coarse.std_error


def test_transaction_costs_shift_mean_hedging_error_downward() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
    )
    frictionless = simulate_delta_hedge(option, "put", paths=1_500, steps=50, rebalance_every=2, seed=8)
    costly = simulate_delta_hedge(
        option,
        "put",
        paths=1_500,
        steps=50,
        rebalance_every=2,
        seed=8,
        transaction_cost_bps=10.0,
    )
    assert costly.mean_error < frictionless.mean_error


def test_realised_vs_implied_volatility_changes_hedging_distribution() -> None:
    option = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.01,
        volatility=0.2,
    )
    matched = simulate_delta_hedge(
        option,
        "call",
        paths=1_500,
        steps=40,
        rebalance_every=2,
        seed=5,
        realised_volatility=0.2,
        implied_volatility=0.2,
    )
    mismatched = simulate_delta_hedge(
        option,
        "call",
        paths=1_500,
        steps=40,
        rebalance_every=2,
        seed=5,
        realised_volatility=0.3,
        implied_volatility=0.2,
    )
    assert abs(matched.mean_error - mismatched.mean_error) > 1e-3
