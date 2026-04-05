"""Serving algorithm implementations."""

from .naive import NaiveAlgorithm
from .static_batch import StaticBatchAlgorithm
from .continuous_batch import ContinuousBatchAlgorithm
from .paged_attention import PagedAttentionAlgorithm
from .quantized import QuantizedAlgorithm
from .speculative import SpeculativeAlgorithm
from .chunked_prefill import ChunkedPrefillAlgorithm

ALGORITHM_MAP: dict[str, type] = {
    "naive": NaiveAlgorithm,
    "static_batch": StaticBatchAlgorithm,
    "continuous_batch": ContinuousBatchAlgorithm,
    "paged_attention": PagedAttentionAlgorithm,
    "quantized": QuantizedAlgorithm,
    "speculative": SpeculativeAlgorithm,
    "chunked_prefill": ChunkedPrefillAlgorithm,
}

__all__ = [
    "NaiveAlgorithm",
    "StaticBatchAlgorithm",
    "ContinuousBatchAlgorithm",
    "PagedAttentionAlgorithm",
    "QuantizedAlgorithm",
    "SpeculativeAlgorithm",
    "ChunkedPrefillAlgorithm",
    "ALGORITHM_MAP",
]
