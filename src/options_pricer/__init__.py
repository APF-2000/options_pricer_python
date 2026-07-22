"""Public package interface for options_pricer."""

from .contracts import EuropeanOption
from .euro_options import make_european_option
from .models import (
    BinomialTreeResult,
    MonteCarloResult,
    OptionGreeks,
    black_scholes_greeks,
    black_scholes_price,
    cox_ross_rubinstein_price,
    european_call_option,
    european_put_option,
    monte_carlo_price,
)
from .visualization import (
    compare_model_prices,
    describe_option,
    plot_price_curve,
    plot_sample_paths,
    render_binomial_tree,
    render_price_table,
    render_sample_paths,
)

__all__ = [
    "BinomialTreeResult",
    "EuropeanOption",
    "MonteCarloResult",
    "OptionGreeks",
    "black_scholes_greeks",
    "black_scholes_price",
    "compare_model_prices",
    "cox_ross_rubinstein_price",
    "describe_option",
    "european_call_option",
    "european_put_option",
    "make_european_option",
    "monte_carlo_price",
    "plot_price_curve",
    "plot_sample_paths",
    "render_binomial_tree",
    "render_price_table",
    "render_sample_paths",
]
