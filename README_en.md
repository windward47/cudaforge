# CudaForge

<div align="center">

**A lightweight CUDA-powered neural network inference engine — zero external dependencies, ONNX-compatible.**

[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C Standard](https://img.shields.io/badge/C-11-A8B9CC?logo=c)](https://en.cppreference.com/w/c/11)
[![C++ Standard](https://img.shields.io/badge/C++-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-808080)](https://github.com/windward47/cudaforge)

[English](README_en.md) | [简体中文](README.md)

</div>

---

CudaForge is a **from-scratch CUDA inference engine** written in C. It loads standard `.onnx` models and runs them on GPU with hand-optimized CUDA kernels — no cuDNN, no TensorRT, no protobuf dependency. Just CMake, a C compiler, and the CUDA Toolkit.

## Why CudaForge?

| Aspect | CudaForge | Typical inference stack |
| --- | --- | --- |
| Dependencies | **Zero** (self-contained protobuf parser) | cuDNN + TensorRT + protobuf + ... |
| Binary size | ~500 KB | 100+ MB |
| Build time | ~10 seconds | Several minutes |
| Hackability | Plain C — read it all in an afternoon | Millions of lines of framework code |

CudaForge is built for **learning, prototyping, and embedded deployment** — not for beating TensorRT on benchmarks. Every CUDA kernel is written by hand and paired with a CPU fallback, so you can trace exactly how data moves from ONNX weights → GPU kernels → inference results.

## Architecture

```text
┌──────────────────────────────────────────────────┐
│  Application Layer                                │
│  ONNX loader · Graph builder · Inference engine   │
├──────────────────────────────────────────────────┤
│  Operator Layer                                   │
│  ReLU · Conv2D · MatMul · Pool · BatchNorm · ...  │
├──────────────────────────────────────────────────┤
│  Platform Layer                                   │
│  CPU abstraction · CUDA memory · Stream management│
└──────────────────────────────────────────────────┘
```

**Strict layering**: Application → Operator → Platform. No reverse or cross-layer dependencies. Every operator ships with both a CUDA kernel and a pure-C fallback for validation.

## Quick Start

### Prerequisites

- CMake ≥ 3.18
- C/C++ compiler (MSVC 2022, GCC, or Clang)
- CUDA Toolkit ≥ 12.0
- Python ≥ 3.8 (optional, for generating test models)

### Build

```bash
git clone https://github.com/windward47/cudaforge.git
cd cudaforge

# CUDA mode (default) — RTX 2050 uses sm_86
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DCUDA_ARCH=86
cmake --build build --config Release -j$(nproc)

# CPU-only fallback
cmake -B build -DENABLE_CUDA=OFF -DENABLE_TESTS=ON
cmake --build build -j$(nproc)
```

### Run

```bash
# Smoke test — prints platform info
./build/Release/cudaforge.exe

# Full test suite (9 test executables)
ctest --test-dir build -C Release -j$(nproc)

# GPU memory safety check
compute-sanitizer ./build/Release/test_onnx.exe
```

Expected output:

```text
CudaForge v0.1.0
Platform: x86_64 (8 cores, 64 B cache line)
CUDA: enabled (1 device(s))
```

## Supported Operators

| Operator | ONNX Op | CPU | CUDA | Notes |
| --- | --- | --- | --- | --- |
| ReLU | `Relu` | ✓ | ✓ | Element-wise, in-place safe |
| Sigmoid | `Sigmoid` | ✓ | ✓ | Element-wise |
| GELU | `Gelu` | ✓ | ✓ | Gaussian error linear unit |
| Conv2D | `Conv` | ✓ | ✓ | Direct + shared memory tiled kernels |
| MatMul/Gemm | `MatMul` / `Gemm` | ✓ | ✓ | Warp-tiled 32×32 kernel |
| MaxPool2D | `MaxPool` | ✓ | ✓ | |
| AvgPool2D | `AveragePool` | ✓ | ✓ | |
| BatchNorm | `BatchNormalization` | ✓ | ✓ | Coalesced + cross-block reduction |

## API at a Glance

### High-level: load an ONNX model and infer

```c
#include "inference_engine.h"
#include "platform.h"

platform_init();
operator_init_all();
#ifdef USE_CUDA
cuda_platform_init(0);
#endif

// Load
inference_session_t* session = inference_session_load("model.onnx");

// Prepare I/O
int64_t shape[] = {1, 4};
tensor_t* input  = tensor_create(DATA_TYPE_F32, 2, shape);
tensor_t* output = tensor_create(DATA_TYPE_F32, 2, shape);
// ... fill input->data ...

// Run (1 = GPU, 0 = CPU)
tensor_t* inputs[]  = { input };
tensor_t* outputs[] = { output };
inference_session_run(session, inputs, outputs, 1);

if (output->data_device) tensor_copy_to_host(output);
// ... read output->data ...

inference_session_destroy(session);
platform_finalize();
```

### Low-level: build a compute graph by hand

```c
graph_t* g = graph_create();
int tid_in  = graph_add_tensor(g, tensor_create(DATA_TYPE_F32, 2, shape));
int tid_out = graph_add_tensor(g, tensor_create(DATA_TYPE_F32, 2, shape));

graph_add_node(g, OP_RELU, 1, (int[]){tid_in}, 1, (int[]){tid_out}, 0, NULL, NULL, 0);
graph_build(g);
graph_execute(g, inputs, outputs, false);  // false = CPU
graph_destroy(g);
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full API reference and design rationale.

## Development

### Project Layout

```text
cudaforge/
├── src/
│   ├── platform/          # Hardware abstraction (CPU + CUDA)
│   │   ├── cpu/           #   x86 CPU impl
│   │   ├── cuda/          #   CUDA device/memory/arena
│   │   └── include/       #   platform.h, cuda_ops.h
│   ├── operator/          # Operator implementations
│   │   ├── blas/          #   MatMul
│   │   ├── nn/            #   ReLU, Conv2D, Pool, BatchNorm, Activations
│   │   └── include/       #   operator.h
│   ├── application/       # Inference engine + model loading
│   │   ├── engine/        #   Graph builder, inference session
│   │   ├── model/         #   ONNX protobuf parser + loader
│   │   └── include/       #   inference_engine.h
│   └── main.c             # Entry point
├── tests/                 # Integration tests + benchmarks
├── docs/                  # Architecture, coding style, CUDA guide
└── third_party/unity/     # Unity test framework (vendored)
```

### Adding a New Operator

Every operator follows the same pattern. Example for a hypothetical `Softmax`:

1. **Create internal header** — `src/operator/nn/softmax_int.h` with `softmax_params_t`
2. **Write CPU fallback** — `src/operator/nn/softmax.c` (pure C, always works)
3. **Write CUDA kernel** — `src/operator/nn/softmax_cuda.cu` (GPU-accelerated)
4. **Register the operator** — add to `operator_registry.c` and `operator.h` enum
5. **Add shape inference** — update `infer_output_shape()` in `onnx_loader.c`
6. **Add ONNX mapping** — map `"Softmax"` to the new op in `onnx_loader.c`
7. **Write tests** — `src/operator/nn/test/test_softmax.c`
8. **Update CMakeLists.txt** — add source files and test target
9. **Run compute-sanitizer** — zero errors required before merge

See [docs/TASK_TEMPLATES/new_operator.md](docs/TASK_TEMPLATES/new_operator.md) for the step-by-step guide.

### Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-operator`)
3. Follow the [coding style](docs/CODING_STYLE.md) and [CUDA guide](docs/CUDA_GUIDE.md)
4. Write tests — CPU fallback + CUDA kernel both tested
5. Run `compute-sanitizer` on your CUDA tests (zero errors)
6. Run the full test suite: `ctest --test-dir build -C Release`
7. Open a PR with a clear description

**Commit style**: one logical change per commit. Use `feat:` / `fix:` / `refactor:` prefixes.

### Build Options

| Option | Description | Default |
| --- | --- | --- |
| `ENABLE_CUDA` | CUDA GPU backend | `ON` |
| `ENABLE_TESTS` | Build test suite | `ON` |
| `ENABLE_AVX2` | AVX2 CPU optimizations | `OFF` |
| `ENABLE_AVX512` | AVX-512 CPU optimizations | `OFF` |
| `ENABLE_OPENMP` | OpenMP parallelism | `ON` |
| `ENABLE_COVERAGE` | Code coverage instrumentation | `OFF` |
| `CUDA_ARCH` | CUDA compute capability (e.g. `86`) | `86` |

## ONNX Compatibility

CudaForge includes a **hand-written protobuf wire-format parser** (~200 lines of C) — no protobuf library needed. It supports:

- Standard `.onnx` files (proto2 and proto3)
- IR version ≤ 13, opset ≤ 11
- `float_data` and `raw_data` weight formats
- Automatic shape inference for all supported operators

**Limitations**: no dynamic shapes, no INT8/FP16 quantization, operators must be from the supported list above.

## FAQ

**Q: Why not just use ONNX Runtime / TensorRT?**
CudaForge is for learning how inference engines work under the hood. Reading its entire codebase takes an afternoon. It's also useful for embedded scenarios where you can't afford 100+ MB of dependencies.

**Q: Can I run ResNet / YOLO / BERT?**
Not yet. Only the 8 operators listed above are implemented. Many real-world models require Reshape, Softmax, Concat, etc. These are on the roadmap.

**Q: Does it support FP16 or INT8?**
Currently FP32 only. Mixed-precision and quantization are planned for future releases.

**Q: How do I debug a CUDA kernel?**
Use `compute-sanitizer` (ships with CUDA Toolkit):

```bash
compute-sanitizer --tool memcheck ./build/Release/test_conv.exe
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unity Test Framework](https://github.com/ThrowTheSwitch/Unity) — vendored for C unit testing
- [ONNX](https://onnx.ai/) — the open neural network exchange format
