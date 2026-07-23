"""Example: implied-volatility smile and surface from synthetic data."""

from pathlib import Path

import pandas as pd

from options_pricer import (
    EuropeanOption,
    black_scholes_price,
    build_volatility_surface,
    implied_volatility_frame,
    plot_volatility_smile,
    plot_volatility_surface,
)


def main() -> None:
    spot = 100.0
    rows = []
    for maturity, base_sigma in ((0.25, 0.18), (0.5, 0.20), (1.0, 0.22)):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            sigma = base_sigma + 0.0008 * abs(strike - 100.0)
            option = EuropeanOption(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=0.03,
                volatility=sigma,
            )
            rows.append(
                {
                    "strike": strike,
                    "maturity": maturity,
                    "option_type": "call",
                    "price": black_scholes_price(option, "call"),
                }
            )
    chain = pd.DataFrame(rows)
    with_iv = implied_volatility_frame(chain, spot=spot, rate=0.03)
    surface = build_volatility_surface(with_iv)
    print(with_iv.head().to_string(index=False))

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    smile_figure, _ = plot_volatility_smile(with_iv, maturity=0.5)
    smile_figure.savefig(output_dir / "implied_vol_smile.png", dpi=160, bbox_inches="tight")
    surface_figure, _ = plot_volatility_surface(surface)
    surface_figure.savefig(output_dir / "implied_vol_surface.png", dpi=160, bbox_inches="tight")
    with_iv.to_csv(output_dir / "synthetic_option_chain_with_iv.csv", index=False)


if __name__ == "__main__":
    main()
