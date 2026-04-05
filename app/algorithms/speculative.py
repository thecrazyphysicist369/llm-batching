"""Speculative Decoding: n-gram based speculation via vLLM engine config."""

import asyncio
import json
import logging
import time

from .base import BaseAlgorithm
from .continuous_batch import ContinuousBatchAlgorithm

logger = logging.getLogger(__name__)


class SpeculativeAlgorithm(BaseAlgorithm):
    """Enable vLLM's n-gram speculative decoding, then run continuous batching.
    N-gram speculation drafts tokens based on prompt n-gram matches, then
    verifies them in a single forward pass. No extra draft model needed.
    """

    name = "speculative"

    @classmethod
    def engine_config(cls):
        from ..model_manager import SPECULATIVE_CONFIG
        return SPECULATIVE_CONFIG

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        model_mgr = self.model_manager

        # Ensure engine is loaded with speculative config
        needed = self.engine_config()
        if model_mgr.current_config() != needed:
            await self.broadcast({
                "type": "model_status",
                "status": "loading",
                "message": "Reloading engine with n-gram speculative decoding...",
            })
            try:
                await model_mgr.load_engine(needed)
            except Exception as exc:
                logger.exception("Failed to load speculative engine")
                await self.broadcast({
                    "type": "model_status",
                    "status": "error",
                    "message": f"Speculative load failed: {exc}",
                })
                self.running = False
                return
            await self.broadcast({
                "type": "model_status",
                "status": "ready",
                "message": "Speculative decoding engine ready (n-gram, K=4).",
            })

        # Run continuous batch — vLLM handles speculative decoding internally
        inner = ContinuousBatchAlgorithm(
            model_manager=self.model_manager,
            broadcast_fn=self.broadcast,
            prompts_file=self.prompts_file,
        )
        inner.auto_mode = self.auto_mode
        inner.name = self.name  # keep "speculative" label

        self._inner = inner
        await inner.run(prompts)
        self.running = False

    def stop(self) -> None:
        super().stop()
        if hasattr(self, "_inner"):
            self._inner.stop()
