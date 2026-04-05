"""Naive Sequential serving: one user at a time via vLLM."""

import asyncio
import logging
import time

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class NaiveAlgorithm(BaseAlgorithm):
    """Process users one at a time, sequentially.
    Only 1 request active in vLLM at a time → low GPU utilization.
    """

    name = "naive"

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        active_uids = sorted(prompts.keys())
        per_user_tps: list[float] = [0.0] * 4
        total_start = time.time()

        for uid in active_uids:
            if not self.running:
                break

            prompt = prompts[uid]
            self.users[uid]["prompt"] = prompt
            self.users[uid]["done"] = False
            self.users[uid]["phase"] = "prefill"
            self.users[uid]["start_time"] = time.time()

            # Mark others as waiting
            for other in active_uids:
                if other != uid and not self.users[other]["done"]:
                    self.users[other]["phase"] = "waiting"

            await self.send_prompt(uid, prompt)

            # Send animation: only this user active
            mem_layout = self.build_memory_layout({uid: 0})
            await self.send_animation_frame(
                self.build_active_users(), mem_layout,
                self.build_compute_slots([uid]),
                current_user_id=uid,
                idle_slots=len(active_uids) - 1,
                total_users=len(active_uids),
            )

            # Switch to decode phase on first token
            async def on_first():
                self.users[uid]["phase"] = "decode"

            result = await self.stream_user(uid, prompt, on_first_token=on_first)

            per_user_tps[uid] = result["tps"]
            self.users[uid]["phase"] = "idle"
            self.users[uid]["done"] = True
            await self.send_done(uid, result["tokens"], result["tps"])

            # Animation update every completion
            mem_layout = self.build_memory_layout({uid: result["tokens"]})
            await self.send_animation_frame(
                self.build_active_users(), mem_layout,
                self.build_compute_slots([]),
                current_user_id=uid,
                idle_slots=len(active_uids) - 1,
                total_users=len(active_uids),
            )

            await asyncio.sleep(0)

        # Benchmark
        total_elapsed = time.time() - total_start
        all_tokens = sum(self.users[uid]["generated_tokens"] for uid in active_uids)
        total_tps = all_tokens / total_elapsed if total_elapsed > 0 else 0.0
        await self.send_benchmark(total_tps, per_user_tps)
        self.running = False
