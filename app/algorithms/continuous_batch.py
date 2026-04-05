"""Continuous Batching: users can finish independently and be replaced."""

import asyncio
import logging
import time

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class ContinuousBatchAlgorithm(BaseAlgorithm):
    """vLLM's natural mode: all requests concurrent, each completes independently.
    Freed slots can be immediately reused. No waiting for the longest sequence.
    """

    name = "continuous_batch"

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        active_uids = sorted(prompts.keys())
        per_user_tps: list[float] = [0.0] * 4
        total_start = time.time()

        # Set up all users
        for uid in active_uids:
            self.users[uid]["prompt"] = prompts[uid]
            self.users[uid]["done"] = False
            self.users[uid]["phase"] = "prefill"
            self.users[uid]["start_time"] = time.time()
            await self.send_prompt(uid, prompts[uid])

        # Send initial animation
        slot_status = ["idle"] * 4
        for uid in active_uids:
            slot_status[uid] = "active"

        await self.send_animation_frame(
            self.build_active_users(),
            self.build_memory_layout({uid: 0 for uid in active_uids}),
            self.build_compute_slots(active_uids),
            slot_status=slot_status,
        )

        async def _run_user(uid: int):
            async def on_first():
                self.users[uid]["phase"] = "decode"

            result = await self.stream_user(uid, prompts[uid], on_first_token=on_first)
            per_user_tps[uid] = result["tps"]
            self.users[uid]["phase"] = "idle"
            self.users[uid]["done"] = True
            slot_status[uid] = "free"

            # Send done IMMEDIATELY (continuous batch = independent completion)
            await self.send_done(uid, result["tokens"], result["tps"])

            # Update animation showing freed slot
            remaining = [u for u in active_uids if not self.users[u]["done"]]
            await self.send_animation_frame(
                self.build_active_users(),
                self.build_memory_layout({u: self.users[u]["generated_tokens"] for u in active_uids}),
                self.build_compute_slots(remaining),
                slot_status=list(slot_status),
            )

        # Launch all concurrently — each finishes independently
        tasks = [asyncio.create_task(_run_user(uid)) for uid in active_uids]
        await asyncio.gather(*tasks)

        # Benchmark
        total_elapsed = time.time() - total_start
        all_tokens = sum(self.users[uid]["generated_tokens"] for uid in active_uids)
        total_tps = all_tokens / total_elapsed if total_elapsed > 0 else 0.0
        await self.send_benchmark(total_tps, per_user_tps)
        self.running = False
