"""Small runtime benchmark suite for the pricing methods."""

from __future__ import annotations

from time import perf_counter

import pandas as pd

from options_pricer import (
    EuropeanOption,
    black_scholes_price,
    compare_monte_carlo_methods,
    cox_ross_rubinstein_price,
    monte_carlo_price,
)


def benchmark_pricers() -> pd.DataFrame:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    rows: list[dict[str, float | str]] = []

    start = perf_counter()
    price = black_scholes_price(option, "call")
    rows.append({"method": "Black-Scholes", "price": price, "runtime_seconds": perf_counter() - start})

    start = perf_counter()
    tree = cox_ross_rubinstein_price(option, "call", steps=500)
    rows.append({"method": "Binomial tree (500 steps)", "price": tree.price, "runtime_seconds": perf_counter() - start})

    for antithetic, control_variate, label in (
        (False, False, "Monte Carlo"),
        (True, False, "Monte Carlo + antithetic"),
        (True, True, "Monte Carlo + antithetic + control variate"),
    ):
        start = perf_counter()
        mc = monte_carlo_price(
            option,
            "call",
            paths=20_000,
            steps=100,
            seed=7,
            antithetic=antithetic,
            control_variate=control_variate,
        )
        rows.append({"method": label, "price": mc.price, "runtime_seconds": perf_counter() - start})
    return pd.DataFrame(rows)


def main() -> None:
    frame = benchmark_pricers()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
