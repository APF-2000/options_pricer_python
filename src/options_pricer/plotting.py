"""Reporting, tables, and plotting helpers."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .binomial import BinomialResult, binomial_convergence_table, cox_ross_rubinstein_price
from .black_scholes import black_scholes_greeks, black_scholes_price
from .hedging import HedgingSimulationResult
from .implied_volatility import build_volatility_surface, implied_volatility_frame
from .instruments import OptionContract, OptionType
from .monte_carlo import MonteCarloResult, compare_monte_carlo_methods, monte_carlo_convergence_table, monte_carlo_price


def _plt():
    import matplotlib.pyplot as plt

    return plt


def describe_option(option: OptionContract, option_type: OptionType) -> str:
    price = black_scholes_price(option, option_type)
    greeks = black_scholes_greeks(option, option_type)
    intrinsic = option.intrinsic_value(option_type)
    time_value = option.time_value(option_type, price)
    return "\n".join(
        [
            f"{option_type.title()} option summary",
            f"Spot={option.spot:.2f} Strike={option.strike:.2f} Tau={option.time_to_maturity:.3f}",
            f"Rate={option.rate:.2%} Dividend yield={option.dividend_yield:.2%} Volatility={option.volatility:.2%}",
            f"Price={price:.4f} Intrinsic={intrinsic:.4f} Time value={time_value:.4f}",
            (
                "Greeks "
                f"Delta={greeks.delta:.4f} Gamma={greeks.gamma:.4f} "
                f"Vega={greeks.vega:.4f} Theta={greeks.theta:.4f} Rho={greeks.rho:.4f}"
            ),
        ]
    )


def price_curve(
    option: OptionContract,
    option_type: OptionType,
    *,
    model: str = "black_scholes",
    spot_min: float | None = None,
    spot_max: float | None = None,
    points: int = 21,
) -> pd.DataFrame:
    if points < 2:
        raise ValueError("points must be at least 2")
    lower = option.strike * 0.6 if spot_min is None else spot_min
    upper = option.strike * 1.4 if spot_max is None else spot_max
    spots = np.linspace(lower, upper, points)
    rows: list[dict[str, float]] = []
    for spot in spots:
        scenario = option.with_spot(float(spot))
        if model == "black_scholes":
            price = black_scholes_price(scenario, option_type)
        elif model == "binomial":
            price = cox_ross_rubinstein_price(scenario, option_type, steps=200).price
        elif model == "monte_carlo":
            price = monte_carlo_price(
                scenario,
                option_type,
                paths=4_000,
                steps=100,
                seed=11,
                antithetic=True,
                control_variate=True,
            ).price
        else:
            raise ValueError("model must be 'black_scholes', 'binomial', or 'monte_carlo'")
        intrinsic = scenario.intrinsic_value(option_type)
        rows.append(
            {
                "spot": float(spot),
                "price": float(price),
                "intrinsic": float(intrinsic),
                "time_value": float(price - intrinsic),
            }
        )
    return pd.DataFrame(rows)


def render_price_table(rows: pd.DataFrame) -> str:
    frame = rows.copy()
    return frame.to_string(
        index=False,
        justify="right",
        formatters={
            "spot": "{:.2f}".format,
            "price": "{:.4f}".format,
            "intrinsic": "{:.4f}".format,
            "time_value": "{:.4f}".format,
        },
    )


def compare_model_prices(
    option: OptionContract,
    option_type: OptionType,
    *,
    binomial_steps: int = 200,
    monte_carlo_paths: int = 8_000,
    monte_carlo_steps: int = 100,
    seed: int = 7,
) -> pd.DataFrame:
    tree = cox_ross_rubinstein_price(option, option_type, steps=binomial_steps)
    mc = monte_carlo_price(
        option,
        option_type,
        paths=monte_carlo_paths,
        steps=monte_carlo_steps,
        seed=seed,
        antithetic=True,
        control_variate=True,
    )
    return pd.DataFrame(
        [
            {
                "model": "Black-Scholes",
                "price": black_scholes_price(option, option_type),
                "notes": "Closed-form benchmark",
            },
            {
                "model": "Binomial tree",
                "price": tree.price,
                "notes": f"{binomial_steps} steps",
            },
            {
                "model": "Monte Carlo",
                "price": mc.price,
                "notes": f"95% CI width {mc.confidence_interval[1] - mc.confidence_interval[0]:.4f}",
            },
        ]
    )


def render_binomial_tree(result: BinomialResult, *, precision: int = 2) -> str:
    if result.stock_tree is None or result.option_tree is None:
        raise ValueError("result does not include tree data; rerun with return_tree=True")
    lines = [
        "Binomial tree view",
        f"Exercise style={result.exercise_style}  Risk-neutral p={result.risk_neutral_probability:.4f}",
    ]
    max_level = len(result.stock_tree) - 1
    exercise_tree = result.early_exercise_tree or [[False] * len(level) for level in result.stock_tree]
    for level in range(max_level + 1):
        indent = " " * (max_level - level) * 2
        nodes = []
        for stock, value, exercised in zip(
            result.stock_tree[level],
            result.option_tree[level],
            exercise_tree[level],
        ):
            tag = "*" if exercised else ""
            nodes.append(f"S={stock:.{precision}f} V={value:.{precision}f}{tag}")
        lines.append(f"t{level:02d} {indent}{' | '.join(nodes)}")
    return "\n".join(lines)


def render_sample_paths(sample_paths: np.ndarray, *, width: int = 10) -> str:
    if sample_paths.size == 0:
        return "No sample paths available."
    stride = max(1, ceil(sample_paths.shape[1] / width))
    lines = ["Monte Carlo sample paths"]
    for index, path in enumerate(sample_paths, start=1):
        condensed = path[::stride]
        if condensed[-1] != path[-1]:
            condensed = np.append(condensed, path[-1])
        lines.append(f"Path {index:02d}: " + " -> ".join(f"{value:.2f}" for value in condensed))
    return "\n".join(lines)


def plot_price_curve(rows: pd.DataFrame, *, title: str = "Option value vs spot"):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(rows["spot"], rows["price"], label="Model price", linewidth=2.5)
    axis.plot(rows["spot"], rows["intrinsic"], label="Intrinsic value", linestyle="--", linewidth=2)
    axis.fill_between(rows["spot"], rows["intrinsic"], rows["price"], alpha=0.18, label="Time value")
    axis.set_title(title)
    axis.set_xlabel("Spot price")
    axis.set_ylabel("Option value")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure, axis


def plot_sample_paths(sample_paths: np.ndarray, *, title: str = "Monte Carlo paths"):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(10, 5))
    for path in sample_paths:
        axis.plot(path, alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Time step")
    axis.set_ylabel("Spot price")
    axis.grid(alpha=0.25)
    return figure, axis


def plot_binomial_convergence(convergence_table: pd.DataFrame):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(convergence_table["steps"], convergence_table["abs_error"], marker="o")
    axis.set_title("Binomial convergence to Black-Scholes")
    axis.set_xlabel("Tree steps")
    axis.set_ylabel("Absolute pricing error")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    return figure, axis


def plot_monte_carlo_convergence(convergence_table: pd.DataFrame):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(convergence_table["paths"], convergence_table["abs_error"], marker="o", label="Absolute error")
    axis.plot(convergence_table["paths"], convergence_table["std_error"], marker="s", label="Standard error")
    axis.set_title("Monte Carlo convergence")
    axis.set_xlabel("Simulation paths")
    axis.set_ylabel("Error")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure, axis


def plot_volatility_smile(chain: pd.DataFrame, *, maturity: float):
    plt = _plt()
    slice_frame = chain.loc[np.isclose(chain["maturity"], maturity)].sort_values("strike")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(slice_frame["strike"], slice_frame["implied_volatility"], marker="o")
    axis.set_title(f"Implied-volatility smile (T={maturity:.2f}y)")
    axis.set_xlabel("Strike")
    axis.set_ylabel("Implied volatility")
    axis.grid(alpha=0.25)
    return figure, axis


def plot_volatility_surface(surface: pd.DataFrame):
    plt = _plt()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    maturities = surface.index.to_numpy(dtype=float)
    strikes = surface.columns.to_numpy(dtype=float)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(strike_grid, maturity_grid, surface.to_numpy(dtype=float), cmap="viridis")
    axis.set_title("Implied-volatility surface")
    axis.set_xlabel("Strike")
    axis.set_ylabel("Maturity (years)")
    axis.set_zlabel("Implied volatility")
    return figure, axis


def plot_hedging_error_distribution(result: HedgingSimulationResult):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(result.hedging_errors, bins=40, alpha=0.8)
    axis.axvline(result.mean_error, color="red", linestyle="--", label="Mean error")
    axis.set_title("Delta-hedging error distribution")
    axis.set_xlabel("Terminal hedging error")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure, axis
