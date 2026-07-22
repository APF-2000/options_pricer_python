"""Text and plotting helpers for understanding option behaviour."""

from __future__ import annotations

from math import ceil

from options_pricer.contracts import EuropeanOption
from options_pricer.models import (
    BinomialTreeResult,
    black_scholes_greeks,
    black_scholes_price,
    cox_ross_rubinstein_price,
    monte_carlo_price,
)


def describe_option(option: EuropeanOption, option_type: str) -> str:
    price = black_scholes_price(option, option_type)
    greeks = black_scholes_greeks(option, option_type)
    intrinsic = option.intrinsic_value(option_type)
    time_value = price - intrinsic
    lines = [
        f"{option_type.title()} option summary",
        f"Spot={option.spot:.2f} Strike={option.strike:.2f} "
        f"Tau={option.time_to_maturity:.2f} Vol={option.volatility:.2%}",
        f"Price={price:.4f} Intrinsic={intrinsic:.4f} Time value={time_value:.4f}",
        "Greeks "
        f"Delta={greeks.delta:.4f} Gamma={greeks.gamma:.4f} "
        f"Vega={greeks.vega:.4f} Theta={greeks.theta:.4f} Rho={greeks.rho:.4f}",
    ]
    return "\n".join(lines)


def price_curve(
    option: EuropeanOption,
    option_type: str,
    *,
    model: str = "black_scholes",
    spot_min: float | None = None,
    spot_max: float | None = None,
    points: int = 11,
) -> list[dict[str, float]]:
    if points < 2:
        raise ValueError("points must be at least 2")

    lower = spot_min if spot_min is not None else option.strike * 0.6
    upper = spot_max if spot_max is not None else option.strike * 1.4
    step = (upper - lower) / (points - 1)
    rows: list[dict[str, float]] = []

    for index in range(points):
        spot = lower + index * step
        scenario = option.with_spot(spot)
        if model == "black_scholes":
            price = black_scholes_price(scenario, option_type)
        elif model == "binomial":
            price = cox_ross_rubinstein_price(scenario, option_type, steps=150).price
        elif model == "monte_carlo":
            price = monte_carlo_price(
                scenario,
                option_type,
                simulations=4_000,
                steps=100,
                seed=11,
            ).price
        else:
            raise ValueError("model must be 'black_scholes', 'binomial', or 'monte_carlo'")

        intrinsic = scenario.intrinsic_value(option_type)
        rows.append(
            {
                "spot": spot,
                "price": price,
                "intrinsic": intrinsic,
                "time_value": price - intrinsic,
            }
        )
    return rows


def render_price_table(rows: list[dict[str, float]]) -> str:
    header = f"{'Spot':>10} {'Price':>12} {'Intrinsic':>12} {'Time Value':>12}"
    separator = "-" * len(header)
    body = [
        f"{row['spot']:>10.2f} {row['price']:>12.4f} "
        f"{row['intrinsic']:>12.4f} {row['time_value']:>12.4f}"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def compare_model_prices(
    option: EuropeanOption,
    option_type: str,
    *,
    binomial_steps: int = 200,
    monte_carlo_simulations: int = 8_000,
    monte_carlo_steps: int = 100,
    seed: int = 7,
) -> str:
    bs_price = black_scholes_price(option, option_type)
    tree = cox_ross_rubinstein_price(option, option_type, steps=binomial_steps)
    mc = monte_carlo_price(
        option,
        option_type,
        simulations=monte_carlo_simulations,
        steps=monte_carlo_steps,
        seed=seed,
    )
    lines = [
        f"{'Model':<20} {'Price':>12} {'Notes':<30}",
        "-" * 62,
        f"{'Black-Scholes':<20} {bs_price:>12.4f} {'Closed form benchmark':<30}",
        f"{'Binomial Tree':<20} {tree.price:>12.4f} {f'{binomial_steps} steps':<30}",
        f"{'Monte Carlo':<20} {mc.price:>12.4f} "
        f"{f'+/- {1.96 * mc.std_error:.4f} (95% CI)':<30}",
    ]
    return "\n".join(lines)


def render_binomial_tree(result: BinomialTreeResult, *, precision: int = 2) -> str:
    lines = [
        "Binomial tree view",
        (
            "Each node shows the stock price (S) and option value (V). "
            f"Risk-neutral p={result.risk_neutral_probability:.4f}"
        ),
    ]
    max_level = len(result.stock_tree) - 1
    for level in range(max_level + 1):
        indent = " " * (max_level - level) * 2
        nodes = []
        for stock, value in zip(result.stock_tree[level], result.option_tree[level]):
            nodes.append(f"S={stock:.{precision}f} V={value:.{precision}f}")
        lines.append(f"t{level:02d} {indent}{' | '.join(nodes)}")
    return "\n".join(lines)


def render_sample_paths(sample_paths: list[list[float]], *, width: int = 10) -> str:
    if not sample_paths:
        return "No paths available."

    max_points = max(len(path) for path in sample_paths)
    stride = max(1, ceil(max_points / width))
    lines = ["Monte Carlo sample paths"]

    for index, path in enumerate(sample_paths, start=1):
        condensed = path[::stride]
        if condensed[-1] != path[-1]:
            condensed.append(path[-1])
        trajectory = " -> ".join(f"{value:.2f}" for value in condensed)
        lines.append(f"Path {index:02d}: {trajectory}")

    return "\n".join(lines)


def plot_price_curve(
    rows: list[dict[str, float]],
    *,
    title: str = "Option price profile",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_price_curve(); install the 'viz' extra"
        ) from exc

    spots = [row["spot"] for row in rows]
    prices = [row["price"] for row in rows]
    intrinsic = [row["intrinsic"] for row in rows]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(spots, prices, label="Model price", linewidth=2.5)
    axis.plot(spots, intrinsic, label="Intrinsic value", linestyle="--", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Spot price")
    axis.set_ylabel("Option value")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure, axis


def plot_sample_paths(sample_paths: list[list[float]], *, title: str = "Monte Carlo paths"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_sample_paths(); install the 'viz' extra"
        ) from exc

    figure, axis = plt.subplots(figsize=(10, 5))
    for path in sample_paths:
        axis.plot(path, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel("Spot price")
    axis.grid(alpha=0.25)
    return figure, axis
