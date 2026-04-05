"""Model loading and lifecycle management via vLLM AsyncLLMEngine."""

import asyncio
import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = str(Path.home() / "models" / "Qwen3.5-4B")


@dataclass(frozen=True)
class EngineConfig:
    """Describes a vLLM engine configuration. Frozen so it's comparable."""

    quantization: str | None = None  # None, "bitsandbytes"
    enable_chunked_prefill: bool = False
    speculative_config: str | None = None  # JSON string or None
    max_num_seqs: int = 4
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 512
    enforce_eager: bool = True  # skip CUDA graph compilation to save VRAM


# Pre-defined configs for each algorithm group
DEFAULT_CONFIG = EngineConfig()
QUANTIZED_CONFIG = EngineConfig(quantization="bitsandbytes")
CHUNKED_CONFIG = EngineConfig(enable_chunked_prefill=True)
SPECULATIVE_CONFIG = EngineConfig(
    speculative_config=json.dumps({
        "method": "ngram",
        "ngram_prompt_lookup_max": 4,
        "num_speculative_tokens": 4,
    })
)


class ModelManager:
    """Manages a vLLM AsyncLLMEngine with config-aware reloading."""

    def __init__(self) -> None:
        self._engine = None
        self._tokenizer = None
        self._current_config: EngineConfig | None = None
        self.model_size_mb: int = 0
        self.model_path: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------
    async def load_engine(self, config: EngineConfig | None = None) -> None:
        """Load (or reload) the vLLM engine with the given config."""
        if config is None:
            config = DEFAULT_CONFIG

        # Unload first if already loaded
        if self._engine is not None:
            await self.unload_engine()

        logger.info(
            "Loading vLLM engine: model=%s, quant=%s, chunked=%s, spec=%s",
            self.model_path, config.quantization,
            config.enable_chunked_prefill, config.speculative_config,
        )

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        kwargs = {
            "model": self.model_path,
            "dtype": "bfloat16" if config.quantization is None else "auto",
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.max_num_seqs,
            "trust_remote_code": True,
            "disable_log_stats": True,
            "enforce_eager": config.enforce_eager,
        }

        if config.quantization:
            kwargs["quantization"] = config.quantization

        if config.enable_chunked_prefill:
            kwargs["enable_chunked_prefill"] = True

        if config.speculative_config:
            kwargs["speculative_config"] = config.speculative_config

        engine_args = AsyncEngineArgs(**kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Load tokenizer separately for chat template support
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._current_config = config

        # Estimate model size
        if config.quantization:
            self.model_size_mb = 2500  # ~2.5 GB for 4-bit
        else:
            self.model_size_mb = 8000  # ~8 GB for bf16

        logger.info(
            "vLLM engine loaded – %d MB (%s)",
            self.model_size_mb,
            config.quantization or "bf16",
        )

    async def unload_engine(self) -> None:
        """Shutdown the engine and free GPU memory."""
        logger.info("Unloading vLLM engine...")
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                pass
            del self._engine
            self._engine = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_config = None
        self.model_size_mb = 0
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    async def reload_if_needed(self, config: EngineConfig) -> bool:
        """Reload engine only if config differs. Returns True if reloaded."""
        if self._current_config == config and self._engine is not None:
            return False
        await self.load_engine(config)
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_engine(self):
        """Return the AsyncLLMEngine instance."""
        return self._engine

    def get_tokenizer(self):
        """Return the tokenizer (for chat template formatting)."""
        return self._tokenizer

    def is_loaded(self) -> bool:
        return self._engine is not None

    def is_quantized(self) -> bool:
        return self._current_config is not None and self._current_config.quantization is not None

    def current_config(self) -> EngineConfig | None:
        return self._current_config
