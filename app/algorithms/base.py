"""Base class for all serving algorithms (vLLM backend)."""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Coroutine

from vllm import SamplingParams

logger = logging.getLogger(__name__)

# Qwen3.5-4B architecture constants (for animation estimation only)
QWEN3_5_4B_NUM_LAYERS = 36
QWEN3_5_4B_NUM_KV_HEADS = 4
QWEN3_5_4B_HEAD_DIM = 128

MAX_NEW_TOKENS = 128


class BaseAlgorithm(ABC):
    """Abstract base for LLM-serving algorithm demonstrations (vLLM backend)."""

    name: str = "base"

    def __init__(
        self,
        model_manager,
        broadcast_fn: Callable[[dict], Coroutine],
        prompts_file: str | Path = "prompts.txt",
    ) -> None:
        self.model_manager = model_manager
        self.broadcast = broadcast_fn
        self.prompts_file = Path(prompts_file)

        # Per-user state
        self.users: dict[int, dict[str, Any]] = {
            uid: self._empty_user_state(uid) for uid in range(4)
        }

        self.running = False
        self.auto_mode = False
        self.user_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    # ------------------------------------------------------------------
    # Engine config (override in subclasses that need special configs)
    # ------------------------------------------------------------------
    @classmethod
    def engine_config(cls):
        """Return the EngineConfig this algorithm needs. Default = bf16."""
        from ..model_manager import DEFAULT_CONFIG
        return DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # User state helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_user_state(uid: int) -> dict[str, Any]:
        return {
            "user_id": uid,
            "prompt": "",
            "generated_tokens": 0,
            "tokens_per_sec": 0.0,
            "phase": "idle",
            "done": True,
            "start_time": 0.0,
        }

    def reset_user(self, uid: int) -> None:
        self.users[uid] = self._empty_user_state(uid)

    def reset_all_users(self) -> None:
        for uid in range(4):
            self.reset_user(uid)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def load_prompts(self) -> list[str]:
        try:
            text = self.prompts_file.read_text(encoding="utf-8")
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
            return prompts
        except FileNotFoundError:
            logger.warning("Prompts file not found: %s", self.prompts_file)
            return [
                "What is photosynthesis?",
                "Explain how gravity works",
                "What causes rainbows to appear?",
                "How do vaccines work?",
            ]

    def pick_random_prompts(self) -> dict[int, str]:
        all_prompts = self.load_prompts()
        if len(all_prompts) < 4:
            all_prompts = all_prompts * 4
        chosen = random.sample(all_prompts, 4)
        return {uid: chosen[uid] for uid in range(4)}

    # ------------------------------------------------------------------
    # vLLM helpers
    # ------------------------------------------------------------------
    def format_prompt(self, prompt: str) -> str:
        """Format a prompt using the model's chat template."""
        tokenizer = self.model_manager.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                return prompt

    def make_sampling_params(self) -> SamplingParams:
        """Create default SamplingParams for generation."""
        return SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,
        )

    async def stream_user(
        self, uid: int, prompt: str,
        on_first_token: Callable | None = None,
    ) -> dict:
        """Stream tokens for one user via vLLM. Returns {tokens, tps, elapsed}.

        This is the core generation helper. All algorithms use this to
        produce tokens and broadcast them to the frontend.
        """
        engine = self.model_manager.get_engine()
        params = self.make_sampling_params()
        formatted = self.format_prompt(prompt)
        request_id = f"{self.name}-u{uid}-{time.time():.4f}"

        prev_len = 0
        token_count = 0
        decode_start = time.time()

        async for output in engine.generate(formatted, params, request_id):
            if not self.running:
                await engine.abort(request_id)
                break

            comp = output.outputs[0]
            new_token_ids = comp.token_ids[prev_len:]
            prev_len = len(comp.token_ids)

            if new_token_ids:
                if token_count == 0 and on_first_token:
                    await on_first_token()

                tokenizer = self.model_manager.get_tokenizer()
                for tid in new_token_ids:
                    token_count += 1
                    self.users[uid]["generated_tokens"] = token_count
                    elapsed = time.time() - decode_start
                    tps = token_count / elapsed if elapsed > 0 else 0.0
                    self.users[uid]["tokens_per_sec"] = tps

                    token_text = tokenizer.decode([tid], skip_special_tokens=False)
                    await self.send_token(uid, token_text, tps)

            if output.finished:
                break

        elapsed = time.time() - decode_start
        tps = token_count / elapsed if elapsed > 0 else 0.0
        return {"tokens": token_count, "tps": tps, "elapsed": elapsed}

    # ------------------------------------------------------------------
    # WebSocket message helpers
    # ------------------------------------------------------------------
    async def send_token(self, user_id: int, token: str, tps: float) -> None:
        await self.broadcast({
            "type": "user_token",
            "user_id": user_id,
            "token": token,
            "tokens_per_sec": round(tps, 2),
        })

    async def send_done(self, user_id: int, total_tokens: int, avg_tps: float) -> None:
        await self.broadcast({
            "type": "user_done",
            "user_id": user_id,
            "total_tokens": total_tokens,
            "avg_tokens_per_sec": round(avg_tps, 2),
            "wait_seconds": 5,
        })

    async def send_prompt(self, user_id: int, prompt: str) -> None:
        await self.broadcast(
            {"type": "user_prompt", "user_id": user_id, "prompt": prompt}
        )

    async def send_animation_frame(
        self,
        active_users: list[dict],
        memory_layout: dict,
        compute_slots: list[dict],
        **extra,
    ) -> None:
        msg = {
            "type": "gpu_animation",
            "algorithm": self.name,
            "active_users": active_users,
            "memory_layout": memory_layout,
            "compute_slots": compute_slots,
        }
        msg.update(extra)
        await self.broadcast(msg)

    async def send_benchmark(
        self, total_tps: float, per_user_tps: list[float]
    ) -> None:
        await self.broadcast({
            "type": "benchmark_result",
            "algorithm": self.name,
            "total_tokens_per_sec": round(total_tps, 2),
            "per_user_tps": [round(t, 2) for t in per_user_tps],
        })

    # ------------------------------------------------------------------
    # Memory estimation (for animation only)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_kv_cache_mb(
        num_tokens: int,
        num_layers: int = QWEN3_5_4B_NUM_LAYERS,
        num_kv_heads: int = QWEN3_5_4B_NUM_KV_HEADS,
        head_dim: int = QWEN3_5_4B_HEAD_DIM,
    ) -> float:
        bytes_total = num_tokens * num_layers * num_kv_heads * head_dim * 2 * 2
        return bytes_total / (1024 ** 2)

    def build_memory_layout(self, per_user_tokens: dict[int, int]) -> dict:
        model_mb = self.model_manager.model_size_mb
        kv_per_user = [
            int(self.estimate_kv_cache_mb(per_user_tokens.get(uid, 0)))
            for uid in range(4)
        ]
        gpu_mem = 12288
        used = model_mb + sum(kv_per_user)
        free = max(0, gpu_mem - used)
        return {
            "model_mb": model_mb,
            "kv_cache_mb": kv_per_user,
            "free_mb": free,
        }

    def build_active_users(self) -> list[dict]:
        return [
            {
                "user_id": uid,
                "phase": self.users[uid]["phase"],
                "kv_cache_blocks": int(
                    self.estimate_kv_cache_mb(self.users[uid]["generated_tokens"])
                ),
                "tokens_generated": self.users[uid]["generated_tokens"],
                "color": self.user_colors[uid],
            }
            for uid in range(4)
        ]

    def build_compute_slots(self, active_uids: list[int] | None = None) -> list[dict]:
        if active_uids is None:
            active_uids = [
                uid for uid in range(4)
                if self.users[uid]["phase"] in ("prefill", "decode")
            ]
        return [
            {"user_id": uid if uid in active_uids else None, "active": uid in active_uids}
            for uid in range(4)
        ]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @abstractmethod
    async def run(self, prompts: dict[int, str]) -> None:
        ...

    def stop(self) -> None:
        self.running = False
        self.auto_mode = False
