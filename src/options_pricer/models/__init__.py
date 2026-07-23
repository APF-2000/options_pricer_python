"""Backward-compatible model exports."""

from options_pricer.binomial import BinomialResult, binomial_convergence_table, cox_ross_rubinstein_price
from options_pricer.black_scholes import (
    black_scholes_greeks,
    black_scholes_greeks_inputs,
    black_scholes_price,
    black_scholes_price_inputs,
    d1,
    d2,
    european_call_option,
    european_put_option,
)
from options_pricer.exotics import asian_option_price
from options_pricer.monte_carlo import (
    MonteCarloResult,
    compare_monte_carlo_methods,
    monte_carlo_convergence_table,
    monte_carlo_price,
    simulate_gbm_paths,
)
from options_pricer.surrogate import (
    AmericanBinomialSurrogate,
    BlackScholesSurrogate,
    SurrogateDataset,
    SurrogateEvaluation,
    SurrogateTrainingConfig,
    SurrogateTrainingSummary,
    generate_american_binomial_surrogate_dataset,
    generate_black_scholes_surrogate_dataset,
    generate_surrogate_dataset,
)

__all__ = [
    "AmericanBinomialSurrogate",
    "BlackScholesSurrogate",
    "BinomialResult",
    "MonteCarloResult",
    "SurrogateDataset",
    "SurrogateEvaluation",
    "SurrogateTrainingConfig",
    "SurrogateTrainingSummary",
    "asian_option_price",
    "binomial_convergence_table",
    "black_scholes_greeks",
    "black_scholes_greeks_inputs",
    "black_scholes_price",
    "black_scholes_price_inputs",
    "compare_monte_carlo_methods",
    "cox_ross_rubinstein_price",
    "d1",
    "d2",
    "european_call_option",
    "european_put_option",
    "generate_american_binomial_surrogate_dataset",
    "generate_black_scholes_surrogate_dataset",
    "generate_surrogate_dataset",
    "monte_carlo_convergence_table",
    "monte_carlo_price",
    "simulate_gbm_paths",
]
