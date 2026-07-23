"""Example: arithmetic-average Asian-option pricing by Monte Carlo."""

from options_pricer import EuropeanOption, asian_option_price


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    result = asian_option_price(option, "call", paths=20_000, steps=100, seed=11)
    print(f"Asian call price: {result.price:.4f}")
    print(f"95% confidence interval: {result.confidence_interval}")


if __name__ == "__main__":
    main()
