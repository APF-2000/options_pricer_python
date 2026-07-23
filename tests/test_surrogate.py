import numpy as np
import pytest

torch = pytest.importorskip("torch")

from options_pricer import (  # noqa: E402
    AmericanBinomialSurrogate,
    BlackScholesSurrogate,
    SurrogateTrainingConfig,
    black_scholes_price_inputs,
    cox_ross_rubinstein_price,
    generate_american_binomial_surrogate_dataset,
    generate_surrogate_dataset,
)
from options_pricer.instruments import AmericanOption  # noqa: E402


def test_surrogate_dataset_generation_has_expected_shape() -> None:
    dataset = generate_surrogate_dataset(256, seed=11)
    assert dataset.features.shape == (256, 7)
    assert dataset.normalized_targets.shape == (256,)
    assert dataset.raw_targets.shape == (256,)
    assert set(dataset.frame["option_type"].unique()) == {"call", "put"}


def test_surrogate_learns_black_scholes_surface_to_reasonable_accuracy() -> None:
    training_data = generate_surrogate_dataset(4_000, seed=17)
    test_data = generate_surrogate_dataset(1_000, seed=18)
    surrogate = BlackScholesSurrogate(hidden_sizes=(96, 96, 48))
    summary = surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=40,
            batch_size=256,
            learning_rate=2e-3,
            validation_fraction=0.15,
            seed=19,
        ),
    )
    evaluation = surrogate.evaluate(test_data)
    assert summary.best_validation_loss < 0.02
    assert evaluation.mae < 0.35
    assert evaluation.rmse < 0.55
    assert evaluation.mean_abs_relative_error < 0.10


def test_surrogate_prediction_tracks_analytical_price_for_known_points() -> None:
    training_data = generate_surrogate_dataset(3_000, seed=21)
    surrogate = BlackScholesSurrogate(hidden_sizes=(64, 64))
    surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=30,
            batch_size=256,
            learning_rate=2e-3,
            validation_fraction=0.2,
            seed=22,
        ),
    )
    predicted = surrogate.predict(
        spot=np.array([90.0, 100.0, 110.0]),
        strike=np.array([100.0, 100.0, 100.0]),
        tau=np.array([1.0, 1.0, 1.0]),
        rate=np.array([0.05, 0.05, 0.05]),
        volatility=np.array([0.2, 0.2, 0.2]),
        dividend_yield=np.zeros(3),
        option_type=np.array(["call", "call", "call"]),
    )
    analytical = np.asarray(
        black_scholes_price_inputs(
            spot=np.array([90.0, 100.0, 110.0]),
            strike=np.array([100.0, 100.0, 100.0]),
            tau=np.array([1.0, 1.0, 1.0]),
            rate=np.array([0.05, 0.05, 0.05]),
            volatility=np.array([0.2, 0.2, 0.2]),
            dividend_yield=np.zeros(3),
            option_type="call",
        ),
        dtype=float,
    )
    assert np.max(np.abs(predicted - analytical)) < 0.65


def test_american_binomial_surrogate_learns_tree_prices_to_reasonable_accuracy() -> None:
    training_data = generate_american_binomial_surrogate_dataset(2_500, seed=31, steps=150)
    test_data = generate_american_binomial_surrogate_dataset(600, seed=32, steps=150)
    surrogate = AmericanBinomialSurrogate(hidden_sizes=(96, 96, 48))
    summary = surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=35,
            batch_size=256,
            learning_rate=2e-3,
            validation_fraction=0.2,
            seed=33,
        ),
    )
    evaluation = surrogate.evaluate(test_data)
    assert summary.best_validation_loss < 0.03
    assert evaluation.mae < 0.65
    assert evaluation.rmse < 0.95
    assert evaluation.mean_abs_relative_error < 0.12


def test_american_binomial_surrogate_tracks_known_tree_prices() -> None:
    training_data = generate_american_binomial_surrogate_dataset(2_200, seed=41, steps=175)
    surrogate = AmericanBinomialSurrogate(hidden_sizes=(96, 96, 48))
    surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=32,
            batch_size=256,
            learning_rate=2e-3,
            validation_fraction=0.2,
            seed=42,
        ),
    )

    spots = np.array([90.0, 100.0, 110.0])
    strikes = np.array([100.0, 100.0, 100.0])
    taus = np.array([1.0, 1.0, 1.0])
    rates = np.array([0.03, 0.03, 0.03])
    volatilities = np.array([0.25, 0.25, 0.25])
    dividends = np.array([0.01, 0.01, 0.01])
    option_types = np.array(["put", "put", "put"])

    predicted = surrogate.predict(
        spot=spots,
        strike=strikes,
        tau=taus,
        rate=rates,
        volatility=volatilities,
        dividend_yield=dividends,
        option_type=option_types,
    )

    tree_prices = np.array(
        [
            cox_ross_rubinstein_price(
                AmericanOption(
                    spot=float(spot),
                    strike=float(strike),
                    maturity=float(tau),
                    rate=float(rate),
                    volatility=float(volatility),
                    dividend_yield=float(dividend_yield),
                ),
                str(option_type),
                steps=175,
                american=True,
            ).price
            for spot, strike, tau, rate, volatility, dividend_yield, option_type in zip(
                spots,
                strikes,
                taus,
                rates,
                volatilities,
                dividends,
                option_types,
                strict=True,
            )
        ],
        dtype=float,
    )
    assert np.max(np.abs(predicted - tree_prices)) < 0.9
