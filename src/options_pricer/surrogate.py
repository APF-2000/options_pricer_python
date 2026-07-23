"""PyTorch surrogate models for option pricing surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .binomial import cox_ross_rubinstein_price
from .black_scholes import black_scholes_price_inputs
from .instruments import AmericanOption, OptionType
from .validation import ensure_option_type


@dataclass(frozen=True)
class SurrogateDataset:
    """Synthetic option dataset used for supervised surrogate training."""

    frame: pd.DataFrame
    features: np.ndarray
    normalized_targets: np.ndarray
    raw_targets: np.ndarray


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    """Training hyperparameters for the surrogate MLP."""

    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    validation_fraction: float = 0.2
    hidden_sizes: tuple[int, ...] = (128, 128, 64)
    seed: int = 7
    device: str = "cpu"


@dataclass(frozen=True)
class SurrogateTrainingSummary:
    """Training summary returned by ``fit``."""

    epochs: int
    best_validation_loss: float
    final_training_loss: float
    device: str
    elapsed_seconds: float


@dataclass(frozen=True)
class SurrogateEvaluation:
    """Evaluation metrics for surrogate predictions."""

    mae: float
    rmse: float
    max_abs_error: float
    mean_abs_relative_error: float
    elapsed_seconds: float


class PricingSurrogateNetwork(nn.Module):
    """Simple multilayer perceptron for a normalized pricing surface."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _device_name(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    option_sign = np.where(frame["option_type"].to_numpy() == "call", 1.0, -1.0)
    return np.column_stack(
        [
            frame["log_moneyness"].to_numpy(dtype=np.float32),
            frame["tau"].to_numpy(dtype=np.float32),
            np.sqrt(frame["tau"].to_numpy(dtype=np.float32)),
            frame["rate"].to_numpy(dtype=np.float32),
            frame["volatility"].to_numpy(dtype=np.float32),
            frame["dividend_yield"].to_numpy(dtype=np.float32),
            option_sign.astype(np.float32),
        ]
    )


def _sample_contract_frame(
    num_samples: int,
    *,
    seed: int,
    strike_range: tuple[float, float],
    log_moneyness_range: tuple[float, float],
    maturity_range: tuple[float, float],
    rate_range: tuple[float, float],
    volatility_range: tuple[float, float],
    dividend_yield_range: tuple[float, float],
) -> pd.DataFrame:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    rng = np.random.default_rng(seed)
    strikes = rng.uniform(*strike_range, size=num_samples)
    log_moneyness = rng.uniform(*log_moneyness_range, size=num_samples)
    spots = strikes * np.exp(log_moneyness)
    tau = rng.uniform(*maturity_range, size=num_samples)
    rates = rng.uniform(*rate_range, size=num_samples)
    volatilities = rng.uniform(*volatility_range, size=num_samples)
    dividend_yields = rng.uniform(*dividend_yield_range, size=num_samples)
    option_type_indicator = rng.integers(0, 2, size=num_samples)
    option_types = np.where(option_type_indicator == 1, "call", "put")

    return pd.DataFrame(
        {
            "spot": spots,
            "strike": strikes,
            "tau": tau,
            "rate": rates,
            "volatility": volatilities,
            "dividend_yield": dividend_yields,
            "option_type": option_types,
            "log_moneyness": log_moneyness,
        }
    )


def _build_dataset(frame: pd.DataFrame, prices: np.ndarray) -> SurrogateDataset:
    normalized_prices = prices / frame["strike"].to_numpy(dtype=np.float64)
    dataset_frame = frame.copy()
    dataset_frame["normalized_price"] = normalized_prices
    dataset_frame["price"] = prices
    return SurrogateDataset(
        frame=dataset_frame,
        features=_feature_matrix(dataset_frame).astype(np.float32),
        normalized_targets=normalized_prices.astype(np.float32),
        raw_targets=prices.astype(np.float32),
    )


def generate_black_scholes_surrogate_dataset(
    num_samples: int,
    *,
    seed: int = 7,
    strike_range: tuple[float, float] = (60.0, 140.0),
    log_moneyness_range: tuple[float, float] = (-0.35, 0.35),
    maturity_range: tuple[float, float] = (1.0 / 365.0, 2.0),
    rate_range: tuple[float, float] = (0.0, 0.08),
    volatility_range: tuple[float, float] = (0.05, 0.70),
    dividend_yield_range: tuple[float, float] = (0.0, 0.05),
) -> SurrogateDataset:
    """Sample a synthetic Black-Scholes surface for supervised learning."""

    frame = _sample_contract_frame(
        num_samples,
        seed=seed,
        strike_range=strike_range,
        log_moneyness_range=log_moneyness_range,
        maturity_range=maturity_range,
        rate_range=rate_range,
        volatility_range=volatility_range,
        dividend_yield_range=dividend_yield_range,
    )

    prices = np.empty(len(frame), dtype=np.float64)
    call_mask = frame["option_type"].to_numpy() == "call"
    put_mask = ~call_mask

    if call_mask.any():
        prices[call_mask] = np.asarray(
            black_scholes_price_inputs(
                spot=frame.loc[call_mask, "spot"].to_numpy(),
                strike=frame.loc[call_mask, "strike"].to_numpy(),
                tau=frame.loc[call_mask, "tau"].to_numpy(),
                rate=frame.loc[call_mask, "rate"].to_numpy(),
                volatility=frame.loc[call_mask, "volatility"].to_numpy(),
                option_type="call",
                dividend_yield=frame.loc[call_mask, "dividend_yield"].to_numpy(),
            ),
            dtype=np.float64,
        )
    if put_mask.any():
        prices[put_mask] = np.asarray(
            black_scholes_price_inputs(
                spot=frame.loc[put_mask, "spot"].to_numpy(),
                strike=frame.loc[put_mask, "strike"].to_numpy(),
                tau=frame.loc[put_mask, "tau"].to_numpy(),
                rate=frame.loc[put_mask, "rate"].to_numpy(),
                volatility=frame.loc[put_mask, "volatility"].to_numpy(),
                option_type="put",
                dividend_yield=frame.loc[put_mask, "dividend_yield"].to_numpy(),
            ),
            dtype=np.float64,
        )

    return _build_dataset(frame, prices)


def generate_american_binomial_surrogate_dataset(
    num_samples: int,
    *,
    seed: int = 7,
    steps: int = 200,
    strike_range: tuple[float, float] = (60.0, 140.0),
    log_moneyness_range: tuple[float, float] = (-0.35, 0.35),
    maturity_range: tuple[float, float] = (1.0 / 365.0, 2.0),
    rate_range: tuple[float, float] = (0.0, 0.08),
    volatility_range: tuple[float, float] = (0.05, 0.70),
    dividend_yield_range: tuple[float, float] = (0.0, 0.08),
) -> SurrogateDataset:
    """Sample a synthetic American-option surface labelled by the CRR tree."""

    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    frame = _sample_contract_frame(
        num_samples,
        seed=seed,
        strike_range=strike_range,
        log_moneyness_range=log_moneyness_range,
        maturity_range=maturity_range,
        rate_range=rate_range,
        volatility_range=volatility_range,
        dividend_yield_range=dividend_yield_range,
    )

    prices = np.fromiter(
        (
            cox_ross_rubinstein_price(
                AmericanOption(
                    spot=float(spot),
                    strike=float(strike),
                    maturity=float(tau),
                    rate=float(rate),
                    volatility=float(volatility),
                    dividend_yield=float(dividend_yield),
                ),
                str(option_type),
                steps=steps,
                american=True,
            ).price
            for spot, strike, tau, rate, volatility, dividend_yield, option_type in zip(
                frame["spot"].to_numpy(),
                frame["strike"].to_numpy(),
                frame["tau"].to_numpy(),
                frame["rate"].to_numpy(),
                frame["volatility"].to_numpy(),
                frame["dividend_yield"].to_numpy(),
                frame["option_type"].to_numpy(),
                strict=True,
            )
        ),
        dtype=np.float64,
        count=len(frame),
    )

    frame["exercise_style"] = "american"
    frame["binomial_steps"] = float(steps)
    return _build_dataset(frame, prices)


def generate_surrogate_dataset(
    num_samples: int,
    **kwargs: object,
) -> SurrogateDataset:
    """Backward-compatible alias for Black-Scholes surrogate data."""

    return generate_black_scholes_surrogate_dataset(num_samples, **kwargs)


class PricingSurfaceSurrogate:
    """Generic MLP surrogate for normalized option pricing surfaces."""

    def __init__(self, hidden_sizes: Sequence[int] = (128, 128, 64)) -> None:
        self.hidden_sizes = tuple(hidden_sizes)
        self.model = PricingSurrogateNetwork(input_dim=7, hidden_sizes=self.hidden_sizes)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None
        self.training_history: list[dict[str, float]] = []
        self.training_summary: SurrogateTrainingSummary | None = None
        self.device_name = "cpu"

    def fit(
        self,
        dataset: SurrogateDataset,
        *,
        config: SurrogateTrainingConfig | None = None,
    ) -> SurrogateTrainingSummary:
        """Fit the surrogate on a pre-labelled synthetic dataset."""

        training_config = config or SurrogateTrainingConfig()
        torch.manual_seed(training_config.seed)
        np.random.seed(training_config.seed)
        self.device_name = _device_name(training_config.device)
        device = torch.device(self.device_name)

        rng = np.random.default_rng(training_config.seed)
        permutation = rng.permutation(len(dataset.features))
        validation_size = max(1, int(training_config.validation_fraction * len(permutation)))
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        if len(training_indices) == 0:
            raise ValueError("validation_fraction is too large for the dataset size")

        train_features = dataset.features[training_indices]
        val_features = dataset.features[validation_indices]
        train_targets = np.log1p(dataset.normalized_targets[training_indices])
        val_targets = np.log1p(dataset.normalized_targets[validation_indices])

        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = np.where(train_features.std(axis=0) < 1e-6, 1.0, train_features.std(axis=0))
        self.target_mean = float(train_targets.mean())
        target_std = float(train_targets.std())
        self.target_std = 1.0 if target_std < 1e-6 else target_std

        train_features_scaled = ((train_features - self.feature_mean) / self.feature_std).astype(np.float32)
        val_features_scaled = ((val_features - self.feature_mean) / self.feature_std).astype(np.float32)
        train_targets_scaled = ((train_targets - self.target_mean) / self.target_std).astype(np.float32)
        val_targets_scaled = ((val_targets - self.target_mean) / self.target_std).astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_features_scaled),
                torch.from_numpy(train_targets_scaled),
            ),
            batch_size=training_config.batch_size,
            shuffle=True,
        )

        self.model = PricingSurrogateNetwork(input_dim=7, hidden_sizes=self.hidden_sizes).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        loss_function = nn.MSELoss()
        best_state: dict[str, torch.Tensor] | None = None
        best_validation_loss = float("inf")
        last_train_loss = float("inf")
        self.training_history = []
        start = perf_counter()

        val_features_tensor = torch.from_numpy(val_features_scaled).to(device)
        val_targets_tensor = torch.from_numpy(val_targets_scaled).to(device)

        for epoch in range(training_config.epochs):
            self.model.train()
            batch_losses: list[float] = []
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = loss_function(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(self.model(val_features_tensor), val_targets_tensor).item()
                )
            last_train_loss = float(np.mean(batch_losses))
            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": last_train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError("training did not produce a valid model state")
        self.model.load_state_dict(best_state)
        self.model = self.model.to("cpu")
        self.device_name = "cpu"
        elapsed_seconds = perf_counter() - start
        self.training_summary = SurrogateTrainingSummary(
            epochs=training_config.epochs,
            best_validation_loss=best_validation_loss,
            final_training_loss=last_train_loss,
            device=device.type,
            elapsed_seconds=elapsed_seconds,
        )
        return self.training_summary

    def _check_fitted(self) -> None:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("surrogate must be fit before prediction")
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("surrogate must be fit before prediction")

    def predict_normalized(self, features: np.ndarray) -> np.ndarray:
        """Predict normalized option prices from feature rows."""

        self._check_fitted()
        scaled_features = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model.eval()
            tensor = torch.from_numpy(scaled_features).to(device)
            predictions_scaled = self.model(tensor).cpu().numpy()
        predictions = predictions_scaled * self.target_std + self.target_mean
        predictions = np.expm1(predictions)
        return np.clip(predictions.astype(np.float64), a_min=0.0, a_max=None)

    def predict(
        self,
        *,
        spot: np.ndarray | float,
        strike: np.ndarray | float,
        tau: np.ndarray | float,
        rate: np.ndarray | float,
        volatility: np.ndarray | float,
        option_type: OptionType | Sequence[str],
        dividend_yield: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Predict raw option prices for scalar or vector inputs."""

        spots = np.atleast_1d(np.asarray(spot, dtype=np.float64))
        strikes = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        tau_values = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        rates = np.atleast_1d(np.asarray(rate, dtype=np.float64))
        volatilities = np.atleast_1d(np.asarray(volatility, dtype=np.float64))
        dividends = np.atleast_1d(np.asarray(dividend_yield, dtype=np.float64))

        broadcasted = np.broadcast_arrays(spots, strikes, tau_values, rates, volatilities, dividends)
        spots, strikes, tau_values, rates, volatilities, dividends = [
            array.astype(np.float64) for array in broadcasted
        ]

        if isinstance(option_type, str):
            ensure_option_type(option_type)
            option_types = np.full(spots.shape, option_type, dtype=object)
        else:
            option_types = np.asarray(option_type, dtype=object)
            option_types = np.broadcast_to(option_types, spots.shape)
            for single_type in np.unique(option_types):
                ensure_option_type(str(single_type))

        frame = pd.DataFrame(
            {
                "spot": spots.ravel(),
                "strike": strikes.ravel(),
                "tau": tau_values.ravel(),
                "rate": rates.ravel(),
                "volatility": volatilities.ravel(),
                "dividend_yield": dividends.ravel(),
                "option_type": option_types.ravel(),
            }
        )
        frame["log_moneyness"] = np.log(frame["spot"] / frame["strike"])
        normalized_predictions = self.predict_normalized(_feature_matrix(frame))
        raw_predictions = normalized_predictions * frame["strike"].to_numpy(dtype=np.float64)
        return raw_predictions.reshape(spots.shape)

    def evaluate(self, dataset: SurrogateDataset) -> SurrogateEvaluation:
        """Evaluate the surrogate on a held-out dataset."""

        start = perf_counter()
        predictions = self.predict_normalized(dataset.features) * dataset.frame["strike"].to_numpy()
        errors = predictions - dataset.raw_targets.astype(np.float64)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_abs_error = float(np.max(np.abs(errors)))
        denominator = np.maximum(np.abs(dataset.raw_targets), 1.0)
        mean_abs_relative_error = float(np.mean(np.abs(errors) / denominator))
        return SurrogateEvaluation(
            mae=mae,
            rmse=rmse,
            max_abs_error=max_abs_error,
            mean_abs_relative_error=mean_abs_relative_error,
            elapsed_seconds=perf_counter() - start,
        )

    def save(self, path: str | Path) -> None:
        """Persist the trained surrogate to disk."""

        self._check_fitted()
        payload = {
            "state_dict": self.model.state_dict(),
            "hidden_sizes": self.hidden_sizes,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "training_history": self.training_history,
            "training_summary": None if self.training_summary is None else self.training_summary.__dict__,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "PricingSurfaceSurrogate":
        """Load a persisted surrogate from disk."""

        payload = torch.load(path, map_location="cpu")
        surrogate = cls(hidden_sizes=tuple(payload["hidden_sizes"]))
        surrogate.model.load_state_dict(payload["state_dict"])
        surrogate.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        surrogate.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        surrogate.target_mean = float(payload["target_mean"])
        surrogate.target_std = float(payload["target_std"])
        surrogate.training_history = list(payload["training_history"])
        if payload["training_summary"] is not None:
            surrogate.training_summary = SurrogateTrainingSummary(**payload["training_summary"])
        return surrogate


class BlackScholesSurrogate(PricingSurfaceSurrogate):
    """MLP surrogate intended for Black-Scholes-labelled datasets."""


class AmericanBinomialSurrogate(PricingSurfaceSurrogate):
    """MLP surrogate intended for American binomial-tree-labelled datasets."""

