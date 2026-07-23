"""Benchmark an American-binomial surrogate against direct CRR pricing."""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

from options_pricer import (
    AmericanBinomialSurrogate,
    SurrogateTrainingConfig,
    cox_ross_rubinstein_price,
    generate_american_binomial_surrogate_dataset,
)
from options_pricer.instruments import AmericanOption


def _median_runtime(function, repeats: int) -> float:
    runtimes: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        runtimes.append(perf_counter() - start)
    return median(runtimes)


def _tree_prices(frame: pd.DataFrame, *, steps: int) -> np.ndarray:
    return np.array(
        [
            cox_ross_rubinstein_price(
                AmericanOption(
                    spot=float(row.spot),
                    strike=float(row.strike),
                    maturity=float(row.tau),
                    rate=float(row.rate),
                    volatility=float(row.volatility),
                    dividend_yield=float(row.dividend_yield),
                ),
                str(row.option_type),
                steps=steps,
                american=True,
            ).price
            for row in frame.itertuples(index=False)
        ],
        dtype=float,
    )


def benchmark_surrogate() -> pd.DataFrame:
    steps = 200
    training_data = generate_american_binomial_surrogate_dataset(12_000, seed=120, steps=steps)
    test_data = generate_american_binomial_surrogate_dataset(3_000, seed=121, steps=steps)

    surrogate = AmericanBinomialSurrogate(hidden_sizes=(128, 128, 64))
    train_start = perf_counter()
    surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=50,
            batch_size=512,
            learning_rate=1e-3,
            validation_fraction=0.2,
            seed=122,
        ),
    )
    training_seconds = perf_counter() - train_start

    surrogate_inputs = {
        "spot": test_data.frame["spot"].to_numpy(),
        "strike": test_data.frame["strike"].to_numpy(),
        "tau": test_data.frame["tau"].to_numpy(),
        "rate": test_data.frame["rate"].to_numpy(),
        "volatility": test_data.frame["volatility"].to_numpy(),
        "dividend_yield": test_data.frame["dividend_yield"].to_numpy(),
        "option_type": test_data.frame["option_type"].to_numpy(),
    }

    tree_prices = _tree_prices(test_data.frame, steps=steps)
    surrogate_prices = surrogate.predict(**surrogate_inputs)

    tree_seconds = _median_runtime(
        lambda: _tree_prices(test_data.frame, steps=steps),
        repeats=5,
    )
    inference_seconds = _median_runtime(
        lambda: surrogate.predict(**surrogate_inputs),
        repeats=20,
    )

    errors = pd.Series(surrogate_prices - tree_prices, dtype=float)
    contract_count = float(len(test_data.frame))
    tree_throughput = contract_count / tree_seconds
    surrogate_throughput = contract_count / inference_seconds
    surrogate_speedup_multiple = tree_seconds / inference_seconds

    return pd.DataFrame(
        [
            {"metric": "batch_contract_count", "value": contract_count},
            {"metric": "binomial_steps", "value": float(steps)},
            {"metric": "training_seconds", "value": training_seconds},
            {"metric": "tree_inference_seconds", "value": tree_seconds},
            {"metric": "surrogate_inference_seconds", "value": inference_seconds},
            {"metric": "tree_contracts_per_second", "value": tree_throughput},
            {"metric": "surrogate_contracts_per_second", "value": surrogate_throughput},
            {"metric": "surrogate_speedup_multiple", "value": surrogate_speedup_multiple},
            {"metric": "surrogate_mae", "value": errors.abs().mean()},
            {"metric": "surrogate_rmse", "value": float(np.sqrt(np.mean(errors.to_numpy() ** 2)))},
            {"metric": "surrogate_max_abs_error", "value": errors.abs().max()},
        ]
    )


def main() -> None:
    frame = benchmark_surrogate()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
