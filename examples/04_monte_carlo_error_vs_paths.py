"""Example: Monte Carlo error and confidence intervals versus path count."""

from pathlib import Path

from options_pricer import EuropeanOption, monte_carlo_convergence_table, plot_monte_carlo_convergence


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    table = monte_carlo_convergence_table(option, "call")
    print(table.to_string(index=False))
    figure, _ = plot_monte_carlo_convergence(table)
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "monte_carlo_convergence.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
