"""PagedAttention: continuous batching with simulated paged KV cache visualization."""

import asyncio
import logging
import time

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)

PAGE_SIZE = 16
TOTAL_PAGES = 200


class PageTable:
    """Simulated paged KV cache memory manager (visualization only)."""

    def __init__(self, total_pages: int = TOTAL_PAGES, page_size: int = PAGE_SIZE):
        self.total_pages = total_pages
        self.page_size = page_size
        self.physical_pages: list[int | None] = [None] * total_pages
        self.page_tables: dict[int, list[int]] = {}

    def allocate_page(self, user_id: int) -> int | None:
        for idx, owner in enumerate(self.physical_pages):
            if owner is None:
                self.physical_pages[idx] = user_id
                if user_id not in self.page_tables:
                    self.page_tables[user_id] = []
                self.page_tables[user_id].append(idx)
                return idx
        return None

    def free_pages(self, user_id: int) -> int:
        freed = 0
        if user_id in self.page_tables:
            for idx in self.page_tables[user_id]:
                self.physical_pages[idx] = None
                freed += 1
            del self.page_tables[user_id]
        return freed

    def get_user_pages(self, user_id: int) -> int:
        return len(self.page_tables.get(user_id, []))

    def get_utilization(self) -> float:
        used = sum(1 for p in self.physical_pages if p is not None)
        return used / self.total_pages if self.total_pages > 0 else 0.0

    def get_fragmentation(self) -> float:
        free_indices = [i for i, p in enumerate(self.physical_pages) if p is None]
        if len(free_indices) <= 1:
            return 0.0
        gaps = sum(
            1 for i in range(1, len(free_indices))
            if free_indices[i] - free_indices[i - 1] > 1
        )
        return gaps / len(free_indices)

    def get_page_map_for_animation(self) -> list[dict]:
        return [
            {"page": idx, "user_id": owner}
            for idx, owner in enumerate(self.physical_pages)
            if owner is not None
        ]


class PagedAttentionAlgorithm(BaseAlgorithm):
    """Continuous batching with paged attention visualization overlay.
    vLLM always uses paged attention internally — this algorithm adds
    a simulated page table visualization for educational purposes.
    """

    name = "paged_attention"

    async def run(self, prompts: dict[int, str]) -> None:
        self.running = True
        self.reset_all_users()

        page_table = PageTable()
        active_uids = sorted(prompts.keys())
        per_user_tps: list[float] = [0.0] * 4
        total_start = time.time()

        # Pre-allocate initial pages per user (simulating prompt KV cache)
        for uid in active_uids:
            self.users[uid]["prompt"] = prompts[uid]
            self.users[uid]["done"] = False
            self.users[uid]["phase"] = "prefill"
            self.users[uid]["start_time"] = time.time()
            await self.send_prompt(uid, prompts[uid])

            # Allocate initial pages for estimated prompt length
            est_prompt_tokens = len(prompts[uid].split()) * 2  # rough estimate
            pages_needed = max(1, (est_prompt_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
            for _ in range(pages_needed):
                page_table.allocate_page(uid)

        await self._send_anim(active_uids, page_table)

        async def _run_user(uid: int):
            prev_tokens = 0

            async def on_first():
                self.users[uid]["phase"] = "decode"

            # Override stream_user to track page allocation during generation
            engine = self.model_manager.get_engine()
            params = self.make_sampling_params()
            formatted = self.format_prompt(prompts[uid])
            request_id = f"{self.name}-u{uid}-{time.time():.4f}"

            prev_len = 0
            token_count = 0
            decode_start = time.time()
            first_token = True

            async for output in engine.generate(formatted, params, request_id):
                if not self.running:
                    await engine.abort(request_id)
                    break

                comp = output.outputs[0]
                new_token_ids = comp.token_ids[prev_len:]
                prev_len = len(comp.token_ids)

                if new_token_ids:
                    if first_token:
                        self.users[uid]["phase"] = "decode"
                        first_token = False

                    tokenizer = self.model_manager.get_tokenizer()
                    for tid in new_token_ids:
                        token_count += 1
                        self.users[uid]["generated_tokens"] = token_count
                        elapsed = time.time() - decode_start
                        tps = token_count / elapsed if elapsed > 0 else 0.0
                        self.users[uid]["tokens_per_sec"] = tps

                        token_text = tokenizer.decode([tid], skip_special_tokens=False)
                        await self.send_token(uid, token_text, tps)

                        # Allocate new page every PAGE_SIZE tokens
                        if token_count % PAGE_SIZE == 0:
                            page_table.allocate_page(uid)

                    # Periodic animation update
                    if token_count % 8 == 0:
                        remaining = [u for u in active_uids if not self.users[u]["done"]]
                        await self._send_anim(remaining, page_table)

                if output.finished:
                    break

            elapsed = time.time() - decode_start
            tps = token_count / elapsed if elapsed > 0 else 0.0
            per_user_tps[uid] = tps

            self.users[uid]["phase"] = "idle"
            self.users[uid]["done"] = True
            await self.send_done(uid, token_count, tps)

            # Free pages
            freed = page_table.free_pages(uid)
            logger.info("User %d done, freed %d pages", uid, freed)

            remaining = [u for u in active_uids if not self.users[u]["done"]]
            await self._send_anim(remaining, page_table)

        # Launch all concurrently
        tasks = [asyncio.create_task(_run_user(uid)) for uid in active_uids]
        await asyncio.gather(*tasks)

        # Benchmark
        total_elapsed = time.time() - total_start
        all_tokens = sum(self.users[uid]["generated_tokens"] for uid in active_uids)
        total_tps = all_tokens / total_elapsed if total_elapsed > 0 else 0.0
        await self.send_benchmark(total_tps, per_user_tps)
        self.running = False

    async def _send_anim(self, active_uids, page_table):
        per_user_tokens = {uid: self.users[uid]["generated_tokens"] for uid in range(4)}

        model_mb = self.model_manager.model_size_mb
        kv_per_user = [
            int(self.estimate_kv_cache_mb(per_user_tokens.get(uid, 0)))
            for uid in range(4)
        ]
        gpu_mem = 12288
        used = model_mb + sum(kv_per_user)
        free = max(0, gpu_mem - used)

        memory_layout = {
            "model_mb": model_mb,
            "kv_cache_mb": kv_per_user,
            "free_mb": free,
            "page_utilization": round(page_table.get_utilization() * 100, 1),
            "page_fragmentation": round(page_table.get_fragmentation() * 100, 1),
            "pages_per_user": [page_table.get_user_pages(uid) for uid in range(4)],
            "page_map": page_table.get_page_map_for_animation(),
            "total_pages": page_table.total_pages,
            "page_size": page_table.page_size,
        }

        await self.send_animation_frame(
            self.build_active_users(), memory_layout,
            self.build_compute_slots(active_uids),
        )
