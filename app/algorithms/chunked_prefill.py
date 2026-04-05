"""Chunked Prefill: enable chunked prefill via vLLM engine config."""

import asyncio
import logging
import time

from .base import BaseAlgorithm
from .continuous_batch import ContinuousBatchAlgorithm

logger = logging.getLogger(__name__)


class ChunkedPrefillAlgorithm(BaseAlgorithm):
    """Enable vLLM's chunked prefill mode, then run continuous batching.
    Chunked prefill splits long prompt processing into chunks, interleaving
    decode steps for other users between chunks to prevent starvation.
    """

    name = "chunked_prefill"

    @classmethod
    def engine_config(cls):
        from ..model_manager import CHUNKED_CONFIG
        return CHUNKED_CONFIG

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        model_mgr = self.model_manager

        # Ensure engine is loaded with chunked prefill config
        needed = self.engine_config()
        if model_mgr.current_config() != needed:
            await self.broadcast({
                "type": "model_status",
                "status": "loading",
                "message": "Reloading engine with chunked prefill...",
            })
            try:
                await model_mgr.load_engine(needed)
            except Exception as exc:
                logger.exception("Failed to load chunked prefill engine")
                await self.broadcast({
                    "type": "model_status",
                    "status": "error",
                    "message": f"Chunked prefill load failed: {exc}",
                })
                self.running = False
                return
            await self.broadcast({
                "type": "model_status",
                "status": "ready",
                "message": "Chunked prefill engine ready.",
            })

        # Run continuous batch — vLLM handles chunked prefill internally
        inner = ContinuousBatchAlgorithm(
            model_manager=self.model_manager,
            broadcast_fn=self.broadcast,
            prompts_file=self.prompts_file,
        )
        inner.auto_mode = self.auto_mode
        inner.name = self.name  # keep "chunked_prefill" label

        self._inner = inner
        await inner.run(prompts)
        self.running = False

    def stop(self) -> None:
        super().stop()
        if hasattr(self, "_inner"):
            self._inner.stop()
