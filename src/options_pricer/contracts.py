"""Domain objects shared by pricing models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EuropeanOption:
    """A plain European option contract specification."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    current_time: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be strictly positive")
        if self.strike <= 0:
            raise ValueError("strike must be strictly positive")
        if self.maturity < self.current_time:
            raise ValueError("maturity must be greater than or equal to current_time")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    @property
    def time_to_maturity(self) -> float:
        return max(self.maturity - self.current_time, 0.0)

    def intrinsic_value(self, option_type: str, *, spot: float | None = None) -> float:
        price = self.spot if spot is None else spot
        if option_type == "call":
            return max(price - self.strike, 0.0)
        if option_type == "put":
            return max(self.strike - price, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def with_spot(self, spot: float) -> "EuropeanOption":
        return EuropeanOption(
            spot=spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            current_time=self.current_time,
            dividend_yield=self.dividend_yield,
        )

    def with_volatility(self, volatility: float) -> "EuropeanOption":
        return EuropeanOption(
            spot=self.spot,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=volatility,
            current_time=self.current_time,
            dividend_yield=self.dividend_yield,
        )
