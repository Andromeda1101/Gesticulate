"""Model trainers for classical and deep baselines."""

from src.models.trainers.classical_trainer import train_model

__all__ = ["train_model", "train_deep_baseline"]


def __getattr__(name: str):
    if name == "train_deep_baseline":
        from src.models.trainers.deep_baseline_trainer import train_deep_baseline

        return train_deep_baseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
