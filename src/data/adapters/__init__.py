"""Dataset-specific ingestion adapters."""

from src.data.adapters.hagrid_adapter import index_samples as index_hagrid_samples
from src.data.adapters.leapgestrecog_adapter import index_samples as index_leapgestrecog_samples

__all__ = [
    "index_hagrid_samples",
    "index_leapgestrecog_samples",
]
