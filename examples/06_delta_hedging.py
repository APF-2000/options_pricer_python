"""Example: delta-hedging error versus rebalancing frequency."""

from pathlib import Path

import pandas as pd

from options_pricer import EuropeanOption, plot_hedging_error_distribution, simulate_delta_hedge


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.01, volatility=0.2)
    rows = []
    for rebalance_every in (1, 5, 21):
        result = simulate_delta_hedge(
            option,
            "call",
            paths=2_000,
            steps=252,
            rebalance_every=rebalance_every,
            seed=7,
        )
        rows.append(
            {
                "rebalance_every": rebalance_every,
                "mean_error": result.mean_error,
                "std_error": result.std_error,
                **result.quantiles,
            }
        )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    detailed = simulate_delta_hedge(option, "call", paths=2_000, steps=252, rebalance_every=5, seed=7)
    figure, _ = plot_hedging_error_distribution(detailed)
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "delta_hedging_error_histogram.png", dpi=160, bbox_inches="tight")
    frame.to_csv(output_dir / "delta_hedging_summary.csv", index=False)


if __name__ == "__main__":
    main()
