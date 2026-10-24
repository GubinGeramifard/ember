# ember

A CPU inference engine for transformer language models, written from scratch in modern C++20.

ember loads a Llama-2 architecture model (RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU feed-forward) and runs autoregressive text generation on the CPU, with a hand-vectorized AVX2 + FMA kernel at its core. No GPU, no external ML dependencies, just the standard library and SIMD intrinsics.

## Status

- **Correctness: verified.** Greedy decoding is **byte-for-byte identical** to the reference implementation across every test prompt, and every optimization (SIMD, each thread count) reproduces the exact same tokens. Correctness is proven, not assumed.
- **Performance** (AMD Ryzen 5 2600, 6 cores / 12 threads, TinyStories models):

  | kernel | 15M tok/s | 110M tok/s |
  |---|---|---|
  | scalar, 1 thread | 75 | 9.6 |
  | AVX2+FMA, 1 thread | 192 (2.6x) | 24.9 (2.6x) |
  | AVX2+FMA, 4 threads | 264 (3.5x) | 34.6 (3.6x) |

- **Key finding:** threading stops helping past ~4 threads because fp32 inference is
  **memory-bandwidth-bound** (every token streams all weights from RAM). The right next
  move is to move *less* data: int8 quantization (Phase 3).

## Build

Requires a C++20 compiler and CMake.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```sh
# ember <model.bin> <tokenizer.bin> [prompt] [steps]
./build/ember models/stories15M.bin models/tokenizer.bin "Once upon a time" 256
```

Models use the Llama-2 (`llama2.c`) checkpoint format. A 15M-parameter TinyStories
model and its tokenizer are small enough to run in well under a second per response.

## Design highlights

- **AVX2 + FMA dot product** with dual accumulators to hide FMA latency, plus a scalar tail. Every matmul reduces to this one hot kernel.
- **KV cache** so each step only computes attention against cached keys and values.
- **Zero-copy weight mapping**: weights are viewed in place from the loaded checkpoint buffer, in the exact on-disk layout.
- **RAII throughout**: the model owns one contiguous buffer; activation scratch and the KV cache are allocated once and reused.

## Roadmap

1. ~~Correct fp32 forward pass, verified against a reference.~~ **Done.**
2. ~~Multithreaded matmul + benchmark harness.~~ **Done** (revealed the memory-bandwidth ceiling).
3. Int8 quantization (smaller, faster) with a measured accuracy/speed tradeoff.
4. Tests, CI, and a writeup of the optimization journey.
