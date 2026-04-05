"""Quantized (4-bit) serving: reload engine in 4-bit then run continuous batching."""

import logging

from .base import BaseAlgorithm
from .continuous_batch import ContinuousBatchAlgorithm

logger = logging.getLogger(__name__)


class QuantizedAlgorithm(BaseAlgorithm):
    """Reload the vLLM engine with bitsandbytes 4-bit quantization,
    then run continuous batching. Demonstrates ~4x memory savings.
    """

    name = "quantized"

    @classmethod
    def engine_config(cls):
        from ..model_manager import QUANTIZED_CONFIG
        return QUANTIZED_CONFIG

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        model_mgr = self.model_manager

        # Ensure engine is loaded with quantized config
        needed = self.engine_config()
        if model_mgr.current_config() != needed:
            await self.broadcast({
                "type": "model_status",
                "status": "loading",
                "message": "Reloading engine in 4-bit quantization...",
            })
            try:
                await model_mgr.load_engine(needed)
            except Exception as exc:
                logger.exception("Failed to load quantized engine")
                await self.broadcast({
                    "type": "model_status",
                    "status": "error",
                    "message": f"Quantized load failed: {exc}",
                })
                self.running = False
                return
            await self.broadcast({
                "type": "model_status",
                "status": "ready",
                "message": f"4-bit engine loaded ({model_mgr.model_size_mb} MB). More memory for KV caches!",
            })

        # Delegate to continuous batch runner
        inner = ContinuousBatchAlgorithm(
            model_manager=self.model_manager,
            broadcast_fn=self.broadcast,
            prompts_file=self.prompts_file,
        )
        inner.auto_mode = self.auto_mode
        inner.name = self.name  # keep "quantized" label

        self._inner = inner
        await inner.run(prompts)
        self.running = False

    def stop(self) -> None:
        super().stop()
        if hasattr(self, "_inner"):
            self._inner.stop()
