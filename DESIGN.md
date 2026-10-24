# Design notes

ember is a CPU inference engine for Llama-2 architecture transformers, written
from scratch in C++20. This document explains how it works and, more
importantly, how it was optimized: the engine went from **10.8 to 98.6 tokens/s
(9.1x)** on a 110M model, and every step of that was driven by measurement, not
guesswork.

## Architecture

A checkpoint is loaded in the `llama2.c` binary format. The forward pass for one
token is the standard decoder-only transformer:

```
embed(token)
for each layer:
    x += Attention(RMSNorm(x))      # RoPE, grouped-query attention, KV cache
    x += FFN(RMSNorm(x))            # SwiGLU: w2( silu(w1 x) * w3 x )
logits = Classifier(RMSNorm(x))
```

The code is split into small pieces: `Config` (the checkpoint header), `Weights`
(views into the weight buffer), `RunState` (activation scratch + KV cache, all
allocated once and reused), `Tokenizer` (SentencePiece-style BPE), and the
`Transformer` that ties them together. Generation is autoregressive with a KV
cache, so step *t* only computes attention against the *t* cached key/value
vectors rather than recomputing the whole prefix.

## The hot path

Essentially all inference time is in one place: matrix-vector products. Each
`matmul` computes, for every output row *i*, the dot product of that weight row
with the activation vector. So the entire engine's speed comes down to a dot
product, and that is the single kernel worth optimizing hard.

## The optimization journey

Benchmarks below are a 110M-parameter TinyStories model on an AMD Ryzen 5 2600
(6 cores / 12 threads, AVX2, dual-channel DDR4), 256 generated tokens.

| configuration | tok/s | speedup | agreement with fp32 |
|---|---|---|---|
| scalar, 1 thread (baseline) | 10.8 | 1.00x | 100% |
| AVX2 + FMA, 1 thread | 34.3 | 3.2x | 100% |
| AVX2 + FMA, 4 threads | 46.8 | 4.3x | 100% |
| **int8 + AVX2, 4 threads** | **98.6** | **9.1x** | **99.6%** |

### 1. SIMD (3.2x)

The dot product is vectorized with AVX2 + FMA, processing 8 floats per
instruction. Two independent accumulators run in parallel to hide FMA latency
(the pipeline can issue the next multiply-add before the previous one retires),
with a scalar tail for the final elements. This alone is a 3.2x win and is
exact, the output is byte-for-byte identical to the scalar path.

### 2. Multithreading (4.3x, then it stalls)

A hand-rolled thread pool splits each matmul's output rows across workers. The
workers are created once and parked on a condition variable; dispatch is
allocation-free (the callable is type-erased by hand into a `void*` + a
trampoline function pointer, rather than a `std::function` that would heap-
allocate on every matmul).

The interesting result is what *didn't* happen: throughput peaks around 4
threads and then regresses. More cores made it slower. The reason is the key
insight of the project, **fp32 inference is memory-bandwidth-bound**. Every
generated token must read all ~440 MB of weights from RAM, and dual-channel
DDR4 saturates well before 12 threads are busy, so extra threads only add
synchronization overhead with no more bandwidth to consume.

### 3. Int8 quantization (9.1x)

Since the bottleneck is bytes moved, not math done, the right move is to move
fewer bytes. Weights are quantized to int8 with group-wise scales (the Q8_0
scheme: each group of 32 values shares one fp32 scale), cutting weight memory
traffic ~4x. Activations are quantized on the fly before each matmul.

AVX2 has no signed-byte multiply, so the int8 dot uses the standard trick:
take `|w|` as unsigned, transfer `w`'s sign onto the activation with
`_mm256_sign_epi8`, then `_mm256_maddubs_epi16` (unsigned x signed) and widen to
int32. This nearly doubled throughput again, for a measured accuracy cost of
**0.4%** (the int8 model picks the same next token as fp32 99.6% of the time).

## Correctness

Optimizing an inference engine is pointless if the output drifts, so correctness
is verified, not assumed:

- The **fp32 path is byte-for-byte identical** to a reference implementation
  (`llama2.c`) under greedy decoding, confirmed by diffing generated text across
  prompts.
- Every **fp32 optimization** (SIMD, each thread count) reproduces the exact
  same tokens; the benchmark asserts this.
- The **int8 path** is scored by teacher-forced next-token agreement against the
  fp32 model (99.6%), which honestly measures the quantization trade rather than
  hiding it.
- A unit suite covers the SIMD kernel vs scalar, thread-pool range coverage, and
  quantization error bounds, and runs in CI on every push.

## What I would do next

- **Flash-attention-style** fused attention to cut activation memory traffic.
- **Int4 quantization** for a further bandwidth cut on larger models.
- **SIMD the RoPE and softmax**, currently scalar (small, but free wins).
- **`mmap` the weights** so multiple processes share one physical copy.
