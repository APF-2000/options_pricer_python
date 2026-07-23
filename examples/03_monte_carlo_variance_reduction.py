"""Example: Monte Carlo variance-reduction comparison."""

from pathlib import Path

from options_pricer import EuropeanOption, compare_monte_carlo_methods


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    comparison = compare_monte_carlo_methods(option, "call", paths=10_000, steps=100, seed=7)
    print(comparison.to_string(index=False))
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "monte_carlo_variance_reduction.csv", index=False)


if __name__ == "__main__":
    main()
