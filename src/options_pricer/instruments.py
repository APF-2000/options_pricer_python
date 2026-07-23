"""Financial instruments and common value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class OptionGreeks:
    """Analytical or finite-difference option greeks."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class OptionContract:
    """Scalar vanilla-option contract parameters."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    current_time: float = 0.0
    dividend_yield: float = 0.0
    exercise_style: ExerciseStyle = "european"

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be strictly positive")
        if self.strike <= 0:
            raise ValueError("strike must be strictly positive")
        if self.maturity < self.current_time:
            raise ValueError("maturity must be greater than or equal to current_time")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.exercise_style not in {"european", "american"}:
            raise ValueError("exercise_style must be 'european' or 'american'")

    @property
    def time_to_maturity(self) -> float:
        return self.maturity - self.current_time

    def intrinsic_value(self, option_type: OptionType, *, spot: float | None = None) -> float:
        price = self.spot if spot is None else spot
        if option_type == "call":
            return max(price - self.strike, 0.0)
        if option_type == "put":
            return max(self.strike - price, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def time_value(
        self,
        option_type: OptionType,
        option_price: float,
        *,
        spot: float | None = None,
    ) -> float:
        return option_price - self.intrinsic_value(option_type, spot=spot)

    def with_spot(self, spot: float) -> "OptionContract":
        return OptionContract(
            spot=spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            current_time=self.current_time,
            dividend_yield=self.dividend_yield,
            exercise_style=self.exercise_style,
        )

    def with_volatility(self, volatility: float) -> "OptionContract":
        return OptionContract(
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=volatility,
            current_time=self.current_time,
            dividend_yield=self.dividend_yield,
            exercise_style=self.exercise_style,
        )

    def with_current_time(self, current_time: float) -> "OptionContract":
        return OptionContract(
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            current_time=current_time,
            dividend_yield=self.dividend_yield,
            exercise_style=self.exercise_style,
        )


@dataclass(frozen=True)
class EuropeanOption(OptionContract):
    """European vanilla option contract."""

    exercise_style: ExerciseStyle = "european"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.exercise_style != "european":
            raise ValueError("EuropeanOption must use exercise_style='european'")


@dataclass(frozen=True)
class AmericanOption(OptionContract):
    """American vanilla option contract."""

    exercise_style: ExerciseStyle = "american"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.exercise_style != "american":
            raise ValueError("AmericanOption must use exercise_style='american'")
