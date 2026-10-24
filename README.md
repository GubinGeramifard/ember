# ember

[![ci](https://github.com/GubinGeramifard/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/GubinGeramifard/ember/actions/workflows/ci.yml)

A CPU inference engine for transformer language models, written from scratch in modern C++20.

ember loads a Llama-2 architecture model (RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU feed-forward) and runs autoregressive text generation on the CPU, with a hand-vectorized AVX2 + FMA kernel at its core. No GPU, no external ML dependencies, just the standard library and SIMD intrinsics.

## Status

- **Correctness: verified.** The fp32 path is **byte-for-byte identical** to the reference
  implementation across every test prompt, and every fp32 optimization reproduces the exact
  same tokens. The int8 path agrees with the fp32 model's next-token prediction **99.6%** of
  the time. Correctness is measured, not assumed.
- **Performance, the full optimization journey** (AMD Ryzen 5 2600, 6c/12t, TinyStories 110M):

  | configuration | tok/s | speedup | agree with fp32 |
  |---|---|---|---|
  | scalar, 1 thread (baseline) | 10.8 | 1.00x | 100% |
  | AVX2 + FMA, 1 thread | 34.3 | 3.2x | 100% |
  | AVX2 + FMA, 4 threads | 46.8 | 4.3x | 100% |
  | **int8 + AVX2, 4 threads** | **98.6** | **9.1x** | **99.6%** |

- **The insight that drove it:** multithreading plateaus around 4 threads because fp32
  inference is **memory-bandwidth-bound**, every token streams all weights from RAM. The
  biggest win therefore came not from more cores but from moving *less* data: int8
  quantization cut weight traffic ~4x and nearly doubled throughput again, for a 0.4%
  accuracy cost. Measure, find the real bottleneck, attack that.

## Build

Requires a C++20 compiler and CMake.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```sh
# ember <model.bin> <tokenizer.bin> [prompt] [steps] [threads] [q]
./build/ember models/stories15M.bin models/tokenizer.bin "Once upon a time" 256

# int8-quantized (pass 'q' as the last argument)
./build/ember models/stories110M.bin models/tokenizer.bin "Once upon a time" 256 0 q

# reproduce the benchmark table
./build/bench models/stories110M.bin models/tokenizer.bin 256
```

Models use the Llama-2 (`llama2.c`) checkpoint format. A 15M-parameter TinyStories
model and its tokenizer are small enough to run in well under a second per response.

## Design highlights

- **AVX2 + FMA dot product** with dual accumulators to hide FMA latency, plus a scalar tail. Every matmul reduces to this one hot kernel.
- **Hand-rolled thread pool** with persistent workers and allocation-free dispatch (manual type erasure instead of `std::function`); the calling thread participates as a worker.
- **Int8 group quantization (Q8_0)** with per-group fp32 scales, dynamic activation quantization, and a signed AVX2 int8 dot (`_mm256_sign_epi8` + `maddubs`). The fp32 weights are freed after quantizing, so only the int8 copy stays resident.
- **KV cache** so each step only computes attention against cached keys and values.
- **RAII throughout**: the model owns its buffers; activation scratch and the KV cache are allocated once and reused.

See [DESIGN.md](DESIGN.md) for the architecture and the full optimization journey.

## Roadmap

1. ~~Correct fp32 forward pass, verified against a reference.~~ **Done.**
2. ~~Multithreaded matmul + benchmark harness.~~ **Done** (revealed the memory-bandwidth ceiling).
3. ~~Int8 quantization with a measured accuracy/speed tradeoff.~~ **Done** (9.1x total, 99.6% agreement).
4. Tests, CI, and a writeup of the optimization journey.
