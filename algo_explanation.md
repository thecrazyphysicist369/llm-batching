# How a Single LLM Serves Multiple Users Simultaneously

> **A deep technical guide for engineers of all backgrounds.**
> This document explains every algorithm implemented in the [LLM Batching Dashboard](https://github.com/thecrazyphysicist369/llm-batching) — from the simplest sequential approach to production-grade systems like vLLM.

---

## Table of Contents

1. [The Fundamental Problem](#1-the-fundamental-problem)
2. [Naive Sequential Processing](#2-naive-sequential-processing)
3. [Static Batching](#3-static-batching)
4. [Continuous Batching](#4-continuous-batching)
5. [The KV Cache Problem](#5-the-kv-cache-problem)
6. [PagedAttention](#6-pagedattention)
7. [Quantization](#7-quantization)
8. [Speculative Decoding](#8-speculative-decoding)
9. [Chunked Prefill](#9-chunked-prefill)
10. [The Full Stack](#10-the-full-stack-in-a-modern-serving-engine)

---

## 1. The Fundamental Problem

Imagine you have a restaurant with one very expensive chef (the GPU) and one very complex recipe book (the LLM weights). The recipe book is so large it takes up most of your kitchen counter (GPU memory). Now 100 customers walk in, all wanting different dishes.

**You cannot clone the chef or the recipe book.** You have one GPU, one copy of the model weights. The question is: how do you serve everyone efficiently?

### Why This Is Hard: The Numbers

A modern LLM like Llama-70B in fp16 precision occupies **140 GB** of VRAM just for the model weights. On top of that, every user needs temporary scratch space called the **KV cache** (explained in [Section 5](#5-the-kv-cache-problem)). A single user with a 4096-token conversation requires roughly **8 GB** of KV cache for a 70B model.

```
Model weights:     140 GB (fixed, shared by all users)
KV cache per user:   8 GB (per user, grows with conversation length)
Total for 10 users: 140 + 80 = 220 GB
Total for 100 users: 140 + 800 = 940 GB  ← way beyond any single GPU
```

Even our smaller Qwen3.5-4B model at bf16 occupies ~8 GB on a 12 GB GPU, leaving only ~4 GB for everything else. Every algorithm in this project represents a different strategy for managing this extreme memory and compute pressure.

### The Two Phases of LLM Generation

Every request goes through two fundamentally different phases:

```
┌──────────────────────────────────────────────────────────────┐
│                        PREFILL PHASE                         │
│                                                              │
│  Input: "What is the capital of France?"                     │
│                                                              │
│  Process ALL input tokens in ONE parallel forward pass       │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐               │
│  │What │ is  │ the │capi-│ tal │ of  │Fran-│               │
│  │     │     │     │     │     │     │ ce? │  → All at once │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘               │
│                                                              │
│  Bottleneck: COMPUTE (lots of math, high arithmetic intensity)│
│  Duration: Fast (one big parallel operation)                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                        DECODE PHASE                          │
│                                                              │
│  Generate tokens ONE AT A TIME, autoregressively:            │
│                                                              │
│  Step 1: → "The"                                             │
│  Step 2: → "capital"                                         │
│  Step 3: → "of"                                              │
│  Step 4: → "France"                                          │
│  Step 5: → "is"                                              │
│  Step 6: → "Paris"                                           │
│  Step 7: → "." → DONE                                        │
│                                                              │
│  Bottleneck: MEMORY BANDWIDTH (moving entire model weights   │
│  through the GPU for every single token generated)           │
│  Duration: Slow (one step per token, sequential)             │
└──────────────────────────────────────────────────────────────┘
```

This distinction — **prefill is compute-bound, decode is memory-bandwidth-bound** — is the root cause of why different serving algorithms exist. The GPU hardware is underutilized during decode because you're moving gigabytes of weights through memory just to produce one tiny token.

---

## 2. Naive Sequential Processing

**The simplest possible approach.** Process one user at a time, start to finish.

```
Timeline:
─────────────────────────────────────────────────────────►  time

User 0: [████████████████████]
User 1:                       [████████████████]
User 2:                                         [███████████████████████]
User 3:                                                                  [██████████]

GPU:    ████░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░████░░░░░░░░░
        ↑ doing work          ↑ idle between   ↑ doing work           ↑ etc.
```

### Why It Is Terrible

During the decode phase (which is most of the time), the GPU is doing this:

1. Load the **entire model** (8 GB) from GPU memory into compute cores
2. Multiply it against **one token's** activations (a few KB)
3. Get **one output token**
4. Repeat

The ratio of useful computation to memory traffic is absurdly low. This is called **low arithmetic intensity**. The GPU's thousands of CUDA cores sit idle waiting for data to arrive from memory.

```
Arithmetic Intensity during decode:

  FLOPs per byte loaded ≈ 2 × d_model / bytes_per_param
  For Qwen3.5-4B (bf16): ≈ 2 × 2560 / 2 = 2560 FLOPs/byte

  Sounds high? But the GPU can do 100+ TFLOPS while memory bandwidth
  is ~500 GB/s. The compute-to-bandwidth ratio of the hardware is:

  100 TFLOPS / 500 GB/s = 200,000 FLOPs/byte (hardware ratio)
  vs. 2,560 FLOPs/byte (actual workload)

  → GPU compute is ~98.7% idle during single-user decode!
```

**GPU Utilization: ~1-5%** of theoretical compute throughput.

### When You Might Actually Use This

- **Debugging** — simple control flow, easy to reason about
- **Single-user CLI chatbots** — no concurrency needed
- **Baseline measurement** — to quantify how much better other algorithms are

---

## 3. Static Batching

**The first fix:** instead of processing one user at a time, group multiple requests into a single GPU operation.

```
Batch all 4 users into one forward pass:

  ┌─────────────────────────────────┐  ← padded to max length
  │ User 0: "What is..." [pad][pad]│
  │ User 1: "Explain..."  [pad]    │
  │ User 2: "How does..." [pad]    │
  │ User 3: "Tell me about..."     │  ← longest sequence
  └─────────────────────────────────┘
               ↓
         ONE forward pass
         (matrix multiply is now [4 × seq_len] instead of [1 × seq_len])
```

### How It Works

Instead of running the model 4 times (once per user), you stack all inputs into a single tensor and run the model once. The matrix multiplications become larger, which makes better use of the GPU's parallel compute cores.

```
Single user:    Weight [d_model × d_model]  ×  Input [1 × d_model]     = Output [1 × d_model]
Batch of 4:     Weight [d_model × d_model]  ×  Input [4 × d_model]     = Output [4 × d_model]

Same weight matrix, but 4× more useful work per memory load!
```

### The Padding Problem

All sequences in a batch must have the same length (GPUs need rectangular tensors). Shorter sequences get **padded** with dummy tokens.

```
Step 1:   [A_tok1, B_tok1, C_tok1, D_tok1]   ← all active ✓
Step 7:   [A_tok7, B_DONE, C_tok7, D_tok7]   ← B finished, but slot wastes compute
Step 10:  [A_tok10, ████, C_DONE, D_tok10]   ← B and C wasting compute
Step 15:  [A_tok15, ████, █████, D_tok15]    ← only A and D doing useful work
Step 20:  [A_DONE, █████, █████, D_tok20]    ← only D doing useful work
Step 25:  [██████, █████, █████, D_DONE]     ← finally done

  Red blocks = WASTED compute (padding)
  The batch waited for User D (the slowest) to finish.
  Users A, B, C were blocked from new work the entire time.
```

### Waste Calculation Example

If users generate 25, 7, 10, and 25 tokens respectively, and the batch is padded to 25:

```
Useful tokens:  25 + 7 + 10 + 25 = 67
Total computed: 25 × 4             = 100
Waste:          100 - 67           = 33 tokens (33% wasted!)
```

The waste gets worse with more variance in sequence lengths and larger batch sizes.

---

## 4. Continuous Batching

**The breakthrough that powers every modern LLM serving engine** (vLLM, TGI, TensorRT-LLM, SGLang).

Instead of batching at the **request level** (one batch per group of complete requests), batch at the **iteration level** (one batch per single decoding step).

```
Step 1:  Batch = [A_tok1, B_tok1, C_tok1, D_tok1]   → forward pass
Step 2:  Batch = [A_tok2, B_tok2, C_tok2, D_tok2]   → forward pass
...
Step 7:  B finishes! → IMMEDIATELY remove B, insert new user E
Step 8:  Batch = [A_tok8, C_tok8, D_tok8, E_tok1]   → no idle slots!
...
Step 10: C finishes! → Insert F
Step 11: Batch = [A_tok11, D_tok11, E_tok4, F_tok1]  → still full!
```

### Why This Is Better — A Visual Comparison

```
STATIC BATCHING:
─────────────────────────────────────────────────────────►
User A: ████████████████████████████████████████████████
User B: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (padding waste)
User C: ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (padding waste)
User D: ████████████████████████████████████████████████
        │←───── entire batch must finish before new work ──────→│

CONTINUOUS BATCHING:
─────────────────────────────────────────────────────────►
User A: ████████████████████████████████████████████████
User B: ████████│ ← freed immediately
User E:         ████████████████████████████████████████  ← fills B's slot
User C: ██████████████████│ ← freed immediately
User F:                   ██████████████████████████████  ← fills C's slot
User D: ████████████████████████████████████████████████
        │←── GPU is ALWAYS doing useful work ──────────────────→│
```

### The Scheduler

The scheduler is the brain of continuous batching. At every decoding step, it:

1. **Checks** which requests just finished → removes them from the active batch
2. **Checks** the waiting queue → inserts new requests if there's room
3. **Assembles** the batch tensor for the next forward pass
4. **Optionally preempts** — if memory is tight, it can evict a low-priority request's KV cache to CPU RAM and restore it later

```
Scheduler decision at each step:

  Available GPU memory blocks: 50
  Active requests: [A (using 12 blocks), C (using 8 blocks), D (using 15 blocks)]
  Waiting queue: [E, F, G]

  Decision: E needs ~10 blocks for prefill. 50 - 35 = 15 free blocks. ✓ Admit E.
  Next step: F needs ~20 blocks. 50 - 45 = 5 free blocks. ✗ Wait.
```

### Performance Impact

In our dashboard benchmarks, continuous batching typically achieves **3-4x higher aggregate throughput** than naive sequential, and **10-30% higher** than static batching, because:

- No padding waste (every computation is useful)
- GPU slots never sit idle (new requests fill freed slots instantly)
- Memory is used more efficiently (only allocated while a request is active)

---

## 5. The KV Cache Problem

Before understanding PagedAttention, you need to understand **why memory is the bottleneck**.

### What Is the KV Cache?

In a transformer, the attention mechanism computes:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

Where:
- **Q** (Query) = "What am I looking for?" (current token)
- **K** (Key) = "What information does each past token offer?" (all previous tokens)
- **V** (Value) = "What is the actual content of each past token?" (all previous tokens)

During autoregressive decoding, **K and V for all previous tokens don't change** — only the current token's Q, K, V are new. So we cache the K and V matrices to avoid recomputing them every step.

```
Token 1: Compute K₁, V₁. Cache them.
Token 2: Compute K₂, V₂. Cache them. Use [K₁,K₂] and [V₁,V₂] for attention.
Token 3: Compute K₃, V₃. Cache them. Use [K₁,K₂,K₃] and [V₁,V₂,V₃].
...
Token N: Only compute K_N, V_N. Reuse all cached K₁..K_{N-1}, V₁..V_{N-1}.
```

### How Big Is It?

```
KV cache per token per layer = 2 × num_kv_heads × head_dim × bytes_per_element
                                ↑ K and V       ↑ attention heads   ↑ bf16 = 2 bytes

For Qwen3.5-4B:
  Per token per layer: 2 × 4 × 128 × 2 = 2,048 bytes = 2 KB
  Per token (all 36 layers): 2 KB × 36 = 72 KB
  Per user (512 token context): 72 KB × 512 = 36 MB

For Llama-70B:
  Per token per layer: 2 × 8 × 128 × 2 = 4,096 bytes = 4 KB
  Per token (all 80 layers): 4 KB × 80 = 320 KB
  Per user (4096 token context): 320 KB × 4096 = 1.3 GB
  100 users: 130 GB (!) — more than the model itself
```

### The Memory Fragmentation Problem

Without careful management, KV cache allocation looks like this:

```
GPU Memory (contiguous allocation):

[Model Weights ~8GB][ User A KV ][  Free  ][ User B KV ][ User C KV ][Free][Free]
                                 ↑ gap                              ↑ gaps

User D arrives needing 200 MB contiguous. Total free = 300 MB.
But the largest contiguous free block is only 150 MB!
→ ALLOCATION FAILS despite having enough total memory.

This is EXTERNAL FRAGMENTATION — same problem that plagues naive
memory allocators in operating systems.
```

---

## 6. PagedAttention

**vLLM's core innovation**, directly inspired by how operating systems manage virtual memory.

### The OS Analogy

In an operating system, programs don't get contiguous physical RAM. Instead:

```
OS Virtual Memory:
  Program A thinks it has addresses 0x0000 - 0xFFFF (contiguous)
  Program B thinks it has addresses 0x0000 - 0xFFFF (contiguous)

  Reality (physical RAM):
  [A_page3][B_page1][A_page1][FREE][B_page2][A_page2][B_page3][FREE]

  Page Table (per program):
  A: virtual_page_0 → physical_page_2
     virtual_page_1 → physical_page_5
     virtual_page_2 → physical_page_0

  B: virtual_page_0 → physical_page_1
     virtual_page_1 → physical_page_4
     virtual_page_2 → physical_page_6
```

PagedAttention does **exactly the same thing** for KV cache:

### How PagedAttention Works

Instead of allocating one big contiguous chunk per user, divide GPU memory into fixed-size **pages** (blocks), typically 16 tokens per page.

```
Physical GPU KV Memory (divided into 16-token pages):

Page: [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]...
Owner: A  B  A  -  B  -  C  A  C  -   B   -   C   A    -   -

User A's page table: logical → physical
  Page 0 → Physical 0  (tokens 0-15)
  Page 1 → Physical 2  (tokens 16-31)
  Page 2 → Physical 7  (tokens 32-47)
  Page 3 → Physical 13 (tokens 48-63)

User B's page table:
  Page 0 → Physical 1  (tokens 0-15)
  Page 1 → Physical 4  (tokens 16-31)
  Page 2 → Physical 10 (tokens 32-47)

User C's page table:
  Page 0 → Physical 6  (tokens 0-15)
  Page 1 → Physical 8  (tokens 16-31)
  Page 2 → Physical 12 (tokens 32-47)
```

### Why This Solves Fragmentation

```
BEFORE (contiguous allocation):
  [AAAAAAA][  free  ][BBBBB][CCCCCCC][free]
  User D needs 6 blocks. Total free = 7. Max contiguous = 5. FAILS!

AFTER (paged allocation):
  [A][B][A][ ][ ][B][C][A][C][ ][B][ ][C][A][ ][ ]
  User D needs 6 pages. Free pages: 4 of them scattered.
  Allocate pages 3, 4, 9, 11, 14, 15 → SUCCESS!
  D doesn't need them to be contiguous.
```

### Advanced: Copy-on-Write Prefix Sharing

If 50 users all share the same system prompt ("You are a helpful assistant..."), their KV cache for those tokens is **identical**. PagedAttention can share those physical pages:

```
System prompt "You are a helpful assistant" → KV pages [0, 1, 2]

User A: [0, 1, 2] → [3]  (shared prefix, then unique page 3)
User B: [0, 1, 2] → [4]  (shared prefix, then unique page 4)
User C: [0, 1, 2] → [5]  (shared prefix, then unique page 5)

3 pages shared instead of 9 allocated = 67% memory savings on the prefix!

When User A diverges from the shared prefix, the system performs
Copy-on-Write: copies the shared page to a new physical page and
modifies only User A's copy.
```

---

## 7. Quantization

**Trade precision for memory.** Instead of storing model weights in 16-bit floating point (2 bytes per parameter), compress them to 4-bit integers (0.5 bytes per parameter).

### The Math

```
Qwen3.5-4B has ~4 billion parameters.

Full precision (bf16):  4B × 2 bytes = 8 GB
4-bit quantized:        4B × 0.5 bytes = 2 GB

Memory saved: 6 GB → now available for KV cache!
```

### How 4-bit Quantization Works (Simplified)

Original weight (bf16): [-0.0234, 0.1567, -0.4891, 0.0012, ...]

1. **Find the range** of values in a group of 128 weights
2. **Map** each value to one of 16 levels (4 bits = 2⁴ = 16 possible values)
3. **Store** a scale factor + zero-point per group (a few extra bytes)
4. At inference time, **dequantize** back to fp16/bf16 for the actual matrix multiply

```
Original values:  [-0.49, -0.35, -0.02, 0.00, 0.12, 0.16, 0.38, 0.49]
Quantize to 4-bit: [  0,    1,     6,    7,    9,   10,   13,   15  ]
Scale: 0.065, Zero: 7

Reconstructed:    [-0.46, -0.39, -0.07, 0.00, 0.13, 0.20, 0.39, 0.52]
                   ↑ close but not exact — this is the quality tradeoff
```

### NF4 (Normal Float 4-bit)

The method used in this project (via bitsandbytes). Instead of uniform quantization levels, NF4 places levels according to a **normal distribution** — more levels near zero (where most weights are) and fewer at the extremes. This reduces quantization error significantly.

### Impact on Serving

```
                    BF16 Model     4-bit Model
Model VRAM:         ~8,000 MB      ~2,000 MB
Free for KV cache:  ~4,000 MB      ~10,000 MB
Max concurrent      ~4 users       ~12+ users
users (est.):       (limited KV)   (much more KV room)
Quality:            Baseline       ~1-2% degradation on benchmarks
```

---

## 8. Speculative Decoding

**The core insight:** in standard autoregressive decoding, you generate tokens one at a time. Each token requires a full forward pass through the model. Speculative decoding generates **K tokens in ~2 forward passes** instead of K forward passes.

### The Restaurant Analogy

Imagine a chef (the large model) who is very accurate but slow. You hire a fast apprentice (the draft model) who guesses what the chef would cook:

1. **Apprentice guesses:** "I think the next 4 dishes will be: Pasta, Salad, Soup, Steak"
2. **Chef verifies all 4 at once:** "Pasta ✓, Salad ✓, Soup ✗ (should be Risotto), Steak — skipped (after the error)"
3. **Result:** 2 correct tokens (Pasta, Salad) + 1 corrected token (Risotto) = 3 tokens from 2 operations (draft + verify)

### How It Works Technically

```
Standard decoding (K tokens = K forward passes):
  Step 1: model(token_N)     → token_{N+1}      [1 forward pass]
  Step 2: model(token_{N+1}) → token_{N+2}      [1 forward pass]
  Step 3: model(token_{N+2}) → token_{N+3}      [1 forward pass]
  Step 4: model(token_{N+3}) → token_{N+4}      [1 forward pass]
  Total: 4 forward passes for 4 tokens

Speculative decoding (K tokens ≈ 2 forward passes):
  Draft:  draft_model generates [d1, d2, d3, d4] greedily    [fast, 4 small passes]
  Verify: model([token_N, d1, d2, d3, d4]) in ONE pass        [1 big forward pass]
          → Check: model's token at position 0 matches d1? ✓ Accept
          → Check: model's token at position 1 matches d2? ✓ Accept
          → Check: model's token at position 2 matches d3? ✗ Reject (use model's token instead)
          → Discard d4 (after first rejection, everything after is discarded)
  Result: 3 tokens from ≈ 2 effective forward passes
```

### N-gram Speculation (Used in This Project)

Instead of a separate draft model (which uses extra VRAM), we use **n-gram matching** from the prompt itself. The idea: if the user's prompt contains "the capital of France is", and the model is generating "The capital of France is...", the n-gram drafter looks at the prompt and predicts "Paris" because it saw similar patterns.

```
Prompt: "France is a country in Europe. The capital of France is Paris.
         Question: What is the capital of France?"

Generating: "The capital of France is..."

N-gram drafter checks the prompt:
  "France is" appeared before, followed by "a" — draft "a"?
  "capital of France is" appeared before, followed by "Paris" — draft "Paris"!

This works especially well for:
  - Repetitive content
  - Structured outputs (JSON, code)
  - Questions that echo the prompt
```

### Speedup Math

```
If acceptance rate = α (fraction of drafts accepted), and K = draft length:

  Expected tokens per verify cycle = 1 + α + α² + ... + α^K ≈ 1/(1-α) for high α

  Example: K=4, α=0.8
  Expected tokens = 1 + 0.8 + 0.64 + 0.512 + 0.41 = 3.36 tokens
  Cost = 1 verify pass + 4 small draft passes ≈ 1.5 effective large passes
  Speedup = 3.36 / 1.5 ≈ 2.2x

  But if α=0.3 (low acceptance):
  Expected tokens = 1 + 0.3 + 0.09 + 0.027 = 1.42 tokens
  Cost ≈ 1.5 passes
  Speedup = 1.42 / 1.5 = 0.95x (actually SLOWER — the overhead isn't worth it)
```

**Speculative decoding only helps when the acceptance rate is high** (~70%+).

---

## 9. Chunked Prefill

### The Problem: Prefill Blocking

Remember that prefill (processing the input prompt) is **compute-bound** — it processes all input tokens in parallel. A long prompt (say, 4096 tokens) requires a massive matrix multiplication that monopolizes the GPU for hundreds of milliseconds.

During this time, **all other users' decode steps are blocked**:

```
WITHOUT chunked prefill:

Time →  ─────────────────────────────────────────────────
User A: [==========PREFILL (2000 tokens)==========][D][D][D][D]
User B: [blocked][blocked][blocked][blocked][blocked][D][D][D]
User C: [blocked][blocked][blocked][blocked][blocked][D][D][D]

Users B and C were waiting to generate their next token,
but User A's huge prefill hogged the GPU for the entire duration.
B and C experience a latency spike (starvation).
```

### The Solution: Chunk the Prefill

Instead of processing all 2000 input tokens at once, split them into chunks (e.g., 256 tokens each) and **interleave** decode steps for other users between chunks:

```
WITH chunked prefill (chunk_size = 256):

Time →  ─────────────────────────────────────────────────
User A: [P_chunk1][P_chunk2][P_chunk3][P_chunk4]...[P_chunk8][D][D]
User B:          [D]       [D]       [D]       [D]          [D][D]
User C:          [D]       [D]       [D]       [D]          [D][D]

Every ~256 tokens of A's prefill, B and C get to generate a token.
Nobody starves. Latency stays consistent for all users.
```

### The Tradeoff

```
Pro: Much lower latency variance (P99 latency improves dramatically)
Pro: Better user experience (no one user blocks everyone)
Con: Total prefill time for User A is slightly longer (overhead of chunking)
Con: More scheduler complexity
```

### When This Matters

Chunked prefill is critical for:
- **Long-context models** (32K, 128K token contexts) — a 128K prefill could block the GPU for seconds
- **Mixed workloads** — some users have short prompts, others have very long ones
- **Latency-sensitive applications** — chat interfaces where every user expects responsive tokens

---

## 10. The Full Stack in a Modern Serving Engine

Here's how all these algorithms fit together in a production system like vLLM:

```
                    ┌──────────────────────────────┐
                    │      Incoming HTTP Requests   │
                    │    (OpenAI-compatible API)     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │      AsyncLLMEngine           │
                    │  (manages request lifecycle)  │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │              SCHEDULER                   │
              │                                          │
              │  ┌─────────────────────────────────┐    │
              │  │  Continuous Batching Logic:      │    │
              │  │  • Which requests join the batch?│    │
              │  │  • Which requests just finished? │    │
              │  │  • Any new requests to admit?    │    │
              │  │  • Need to preempt anyone?       │    │
              │  └─────────────────────────────────┘    │
              │                                          │
              │  ┌─────────────────────────────────┐    │
              │  │  PagedAttention Block Manager:   │    │
              │  │  • Allocate KV cache pages       │    │
              │  │  • Free pages on completion       │    │
              │  │  • Track page tables per request  │    │
              │  │  • Handle prefix sharing (CoW)    │    │
              │  └─────────────────────────────────┘    │
              │                                          │
              │  ┌─────────────────────────────────┐    │
              │  │  Chunked Prefill Scheduler:      │    │
              │  │  • Split long prefills into chunks│    │
              │  │  • Interleave with decode steps   │    │
              │  └─────────────────────────────────┘    │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │         GPU WORKER (per GPU)             │
              │                                          │
              │  Model weights (quantized or fp16/bf16)  │
              │  KV cache pages (managed by block mgr)   │
              │  CUDA kernels (FlashAttention, etc.)     │
              │                                          │
              │  If speculative decoding enabled:         │
              │  ┌─────────────────────────────────┐    │
              │  │  Draft → Verify → Accept/Reject  │    │
              │  └─────────────────────────────────┘    │
              │                                          │
              │  If tensor parallelism enabled:           │
              │  ┌─────────────────────────────────┐    │
              │  │  AllReduce across GPU shards      │    │
              │  └─────────────────────────────────┘    │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │              SAMPLER                      │
              │  • Temperature scaling                   │
              │  • Top-p (nucleus) sampling               │
              │  • Top-k filtering                       │
              │  • Repetition penalty                    │
              │  • Stop token detection                  │
              └────────────────────┬────────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   Streamed Token Output       │
                    │   (SSE / WebSocket)           │
                    └──────────────────────────────┘
```

### What Happens on Each Decoding Step

1. **Scheduler** assembles the batch: which requests are active, which are new, which are done
2. **Block Manager** allocates/frees KV cache pages as needed
3. **GPU Worker** runs one forward pass through the model for the entire batch
4. **Sampler** picks the next token for each request in the batch
5. **Output** streams the new tokens back to each client
6. **Repeat** from step 1

This entire cycle happens in **milliseconds** and repeats thousands of times per second for a batch of requests.

---

## Further Reading

- [vLLM Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca Paper: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [Speculative Decoding Paper: Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Sarathi-Serve: Chunked Prefills for Efficient LLM Inference](https://arxiv.org/abs/2308.16369)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

---

*This document is part of the [LLM Batching Dashboard](https://github.com/thecrazyphysicist369/llm-batching) project.*
