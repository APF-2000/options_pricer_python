"""Example: train a PyTorch surrogate for American binomial-tree pricing."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_pricer import (
    AmericanBinomialSurrogate,
    SurrogateTrainingConfig,
    generate_american_binomial_surrogate_dataset,
)


def main() -> None:
    steps = 200
    training_data = generate_american_binomial_surrogate_dataset(10_000, seed=101, steps=steps)
    test_data = generate_american_binomial_surrogate_dataset(2_500, seed=102, steps=steps)

    surrogate = AmericanBinomialSurrogate(hidden_sizes=(128, 128, 64))
    training_summary = surrogate.fit(
        training_data,
        config=SurrogateTrainingConfig(
            epochs=60,
            batch_size=512,
            learning_rate=1e-3,
            validation_fraction=0.2,
            seed=103,
        ),
    )
    evaluation = surrogate.evaluate(test_data)

    print(training_summary)
    print(evaluation)

    predictions = surrogate.predict_normalized(test_data.features) * test_data.frame["strike"].to_numpy()
    diagnostics = pd.DataFrame(
        {
            "true_price": test_data.raw_targets,
            "predicted_price": predictions,
            "abs_error": abs(predictions - test_data.raw_targets),
        }
    )

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "surrogate_predictions.csv", index=False)

    history = pd.DataFrame(surrogate.training_history)
    history.to_csv(output_dir / "surrogate_training_history.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation loss")
    axes[0].set_title("American tree surrogate training curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    sampled = diagnostics.sample(n=min(1_000, len(diagnostics)), random_state=104)
    axes[1].scatter(sampled["true_price"], sampled["predicted_price"], alpha=0.35, s=12)
    min_price = float(min(sampled["true_price"].min(), sampled["predicted_price"].min()))
    max_price = float(max(sampled["true_price"].max(), sampled["predicted_price"].max()))
    axes[1].plot([min_price, max_price], [min_price, max_price], linestyle="--", color="black")
    axes[1].set_title("Surrogate vs CRR American prices")
    axes[1].set_xlabel(f"True CRR price ({steps} steps)")
    axes[1].set_ylabel("Surrogate price")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "surrogate_model_diagnostics.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
