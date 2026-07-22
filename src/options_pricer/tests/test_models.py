import math
import unittest

from options_pricer import (
    EuropeanOption,
    black_scholes_price,
    compare_model_prices,
    cox_ross_rubinstein_price,
    describe_option,
    european_call_option,
    european_put_option,
    monte_carlo_price,
    render_binomial_tree,
    render_price_table,
    render_sample_paths,
)
from options_pricer.visualization import price_curve


class OptionPricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.option = EuropeanOption(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            current_time=0.0,
            rate=0.05,
            volatility=0.2,
        )

    def test_black_scholes_matches_benchmark_values(self) -> None:
        self.assertAlmostEqual(
            european_call_option(0.2, 1.0, 0.0, 100.0, 100.0, 0.05),
            10.4506,
            places=4,
        )
        self.assertAlmostEqual(
            european_put_option(0.2, 1.0, 0.0, 100.0, 100.0, 0.05),
            5.5735,
            places=4,
        )

    def test_put_call_parity_holds(self) -> None:
        call_price = black_scholes_price(self.option, "call")
        put_price = black_scholes_price(self.option, "put")
        parity_gap = call_price - put_price
        benchmark = self.option.spot - self.option.strike * math.exp(
            -self.option.rate * self.option.time_to_maturity
        )
        self.assertAlmostEqual(parity_gap, benchmark, places=6)

    def test_price_at_expiry_is_intrinsic(self) -> None:
        expired = EuropeanOption(
            spot=120.0,
            strike=100.0,
            maturity=1.0,
            current_time=1.0,
            rate=0.05,
            volatility=0.2,
        )
        self.assertEqual(black_scholes_price(expired, "call"), 20.0)
        self.assertEqual(black_scholes_price(expired, "put"), 0.0)

    def test_zero_volatility_reduces_to_discounted_deterministic_payoff(self) -> None:
        option = EuropeanOption(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            current_time=0.0,
            rate=0.05,
            volatility=0.0,
        )
        expected = max(100.0 * math.exp(0.05) - 100.0, 0.0) * math.exp(-0.05)
        self.assertAlmostEqual(black_scholes_price(option, "call"), expected, places=6)

    def test_binomial_tree_converges_close_to_black_scholes(self) -> None:
        bs_price = black_scholes_price(self.option, "call")
        tree = cox_ross_rubinstein_price(self.option, "call", steps=300)
        self.assertLess(abs(tree.price - bs_price), 0.01)

    def test_monte_carlo_stays_close_to_black_scholes(self) -> None:
        bs_price = black_scholes_price(self.option, "call")
        mc = monte_carlo_price(
            self.option,
            "call",
            simulations=6_000,
            steps=80,
            seed=19,
        )
        self.assertLess(abs(mc.price - bs_price), 0.75)
        self.assertGreater(len(mc.sample_paths), 0)

    def test_visualization_helpers_render_expected_content(self) -> None:
        rows = price_curve(self.option, "call", points=5)
        table = render_price_table(rows)
        self.assertIn("Spot", table)
        self.assertIn("Time Value", table)

        tree = cox_ross_rubinstein_price(self.option, "call", steps=3)
        tree_text = render_binomial_tree(tree)
        self.assertIn("Binomial tree view", tree_text)
        self.assertIn("S=", tree_text)

        mc = monte_carlo_price(self.option, "call", simulations=20, steps=8, seed=5)
        paths_text = render_sample_paths(mc.sample_paths)
        self.assertIn("Monte Carlo sample paths", paths_text)

        summary = describe_option(self.option, "call")
        self.assertIn("Call option summary", summary)

        comparison = compare_model_prices(self.option, "call")
        self.assertIn("Black-Scholes", comparison)
        self.assertIn("Monte Carlo", comparison)


if __name__ == "__main__":
    unittest.main()
