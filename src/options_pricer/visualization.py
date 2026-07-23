"""Backward-compatible re-exports for plotting helpers."""

from .plotting import (
    compare_model_prices,
    describe_option,
    plot_binomial_convergence,
    plot_hedging_error_distribution,
    plot_monte_carlo_convergence,
    plot_price_curve,
    plot_sample_paths,
    plot_volatility_smile,
    plot_volatility_surface,
    price_curve,
    render_binomial_tree,
    render_price_table,
    render_sample_paths,
)

__all__ = [
    "compare_model_prices",
    "describe_option",
    "plot_binomial_convergence",
    "plot_hedging_error_distribution",
    "plot_monte_carlo_convergence",
    "plot_price_curve",
    "plot_sample_paths",
    "plot_volatility_smile",
    "plot_volatility_surface",
    "price_curve",
    "render_binomial_tree",
    "render_price_table",
    "render_sample_paths",
]
