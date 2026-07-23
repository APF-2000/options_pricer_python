"""Example: binomial convergence to Black-Scholes."""

from pathlib import Path

from options_pricer import EuropeanOption, binomial_convergence_table, plot_binomial_convergence


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    table = binomial_convergence_table(option, "call")
    print(table.to_string(index=False))
    figure, _ = plot_binomial_convergence(table)
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "binomial_convergence.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
