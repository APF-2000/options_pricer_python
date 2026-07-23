"""Example: Black-Scholes pricing and analytical greeks."""

from options_pricer import EuropeanOption, black_scholes_greeks, black_scholes_price, describe_option


def main() -> None:
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)
    print(describe_option(option, "call"))
    print(f"Put price: {black_scholes_price(option, 'put'):.4f}")
    print(f"Call greeks: {black_scholes_greeks(option, 'call')}")


if __name__ == "__main__":
    main()
