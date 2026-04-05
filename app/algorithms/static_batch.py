"""Static Batching: all users submitted together, everyone waits for longest."""

import asyncio
import logging
import time

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class StaticBatchAlgorithm(BaseAlgorithm):
    """Submit all requests concurrently via vLLM, but defer completion
    until ALL users finish. Simulates static batch: everyone waits for the slowest.
    """

    name = "static_batch"

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
        mem_layout = self.build_memory_layout({uid: 0 for uid in active_uids})
        await self.send_animation_frame(
            self.build_active_users(), mem_layout,
            self.build_compute_slots(active_uids),
            padded_length=0,
            per_user_actual_tokens=[0] * len(active_uids),
            eos_reached=[False] * len(active_uids),
            active_uids=active_uids,
        )

        # Track per-user completion
        user_results: dict[int, dict] = {}
        eos_reached = [False] * len(active_uids)

        async def _run_user(idx: int, uid: int):
            async def on_first():
                self.users[uid]["phase"] = "decode"

            result = await self.stream_user(uid, prompts[uid], on_first_token=on_first)
            user_results[uid] = result
            eos_reached[idx] = True
            # DON'T send_done yet — must wait for all users (static batch)
            self.users[uid]["phase"] = "waiting"

        # Launch all concurrently
        tasks = [
            asyncio.create_task(_run_user(i, uid))
            for i, uid in enumerate(active_uids)
        ]

        # Wait for ALL to finish
        await asyncio.gather(*tasks)

        # NOW send done for everyone simultaneously
        total_elapsed = time.time() - total_start
        for i, uid in enumerate(active_uids):
            r = user_results.get(uid, {"tokens": 0, "tps": 0})
            per_user_tps[uid] = r["tps"]
            self.users[uid]["phase"] = "idle"
            self.users[uid]["done"] = True
            await self.send_done(uid, r["tokens"], r["tps"])

        # Final animation
        per_user_tokens = {uid: self.users[uid]["generated_tokens"] for uid in active_uids}
        max_tokens = max(per_user_tokens.values()) if per_user_tokens else 0
        await self.send_animation_frame(
            self.build_active_users(),
            self.build_memory_layout(per_user_tokens),
            self.build_compute_slots([]),
            padded_length=max_tokens,
            per_user_actual_tokens=[self.users[uid]["generated_tokens"] for uid in active_uids],
            eos_reached=[True] * len(active_uids),
            active_uids=active_uids,
        )

        # Benchmark
        all_tokens = sum(self.users[uid]["generated_tokens"] for uid in active_uids)
        total_tps = all_tokens / total_elapsed if total_elapsed > 0 else 0.0
        await self.send_benchmark(total_tps, per_user_tps)
        self.running = False
