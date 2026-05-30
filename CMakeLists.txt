cmake_minimum_required(VERSION 3.18)
project(CudaForge VERSION 0.1.0 LANGUAGES C CXX)

# ============================================================
# Options
# ============================================================
option(ENABLE_CUDA    "Enable CUDA backend"           ON)
option(ENABLE_TESTS   "Build tests"                    ON)
option(ENABLE_AVX2    "Enable AVX2 optimizations"      OFF)
option(ENABLE_AVX512  "Enable AVX-512 optimizations"   OFF)
option(ENABLE_OPENMP  "Enable OpenMP parallelism"      ON)
option(ENABLE_COVERAGE "Enable coverage"               OFF)

set(CUDA_ARCH "86" CACHE STRING "CUDA architecture target (e.g. 75, 86, 89)")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# ============================================================
# Compiler warnings
# ============================================================
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /W4 /WX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ============================================================
# CUDA setup
# ============================================================
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# ============================================================
# Platform layer (static library)
# ============================================================
add_library(platform STATIC)

target_sources(platform PRIVATE
    src/platform/cpu/platform_cpu.c
    src/platform/tensor.c
    src/platform/data_type.c
)

if(ENABLE_CUDA)
    target_sources(platform PRIVATE
        src/platform/cuda/cuda_device.cu
        src/platform/cuda/cuda_memory.cu
        src/platform/cuda/cuda_platform_init.cu
        src/platform/cuda/cuda_arena.cu
    )
    target_include_directories(platform PUBLIC src/platform/cuda)
endif()

target_include_directories(platform PUBLIC
    src/platform/include
)

if(ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(platform PUBLIC OpenMP::OpenMP_C)
endif()

# ============================================================
# Operator layer (static library)
# ============================================================
add_library(operator STATIC)

target_sources(operator PRIVATE
    src/operator/nn/relu.c
    src/operator/nn/conv.c
    src/operator/nn/activations.c
    src/operator/nn/pooling.c
    src/operator/nn/batchnorm.c
    src/operator/blas/matmul.c
    src/operator/nn/add.c
    src/operator/nn/reshape.c
    src/operator/nn/globalavgpool.c
    src/operator/nn/softmax.c
    src/operator/nn/mul.c
    src/operator/nn/concat.c
    src/operator/nn/resize.c
    src/operator/nn/transpose.c
    src/operator/nn/sub.c
    src/operator/nn/div.c
    src/operator/nn/slice.c
    src/operator/nn/split.c
    src/operator/nn/layernorm.c
    src/operator/nn/gather.c
    src/operator/nn/squeeze_unsqueeze.c
    src/operator/nn/reduce.c
    src/operator/nn/cast.c
    src/operator/nn/argmax.c
    src/operator/nn/mha_fused.c
    src/operator/nn/causal_mask.c
    src/operator/nn/mha_decode.c
    src/operator/nn/rope.c
    src/operator/operator_registry.c
    src/operator/operator_init.c
)

if(ENABLE_CUDA)
    target_sources(operator PRIVATE
        src/operator/nn/relu_cuda.cu
        src/operator/nn/conv_cuda.cu
        src/operator/nn/activations_cuda.cu
        src/operator/nn/pooling_cuda.cu
        src/operator/nn/batchnorm_cuda.cu
        src/operator/nn/add_cuda.cu
        src/operator/nn/reshape_cuda.cu
        src/operator/nn/globalavgpool_cuda.cu
        src/operator/nn/softmax_cuda.cu
        src/operator/nn/mul_cuda.cu
        src/operator/nn/concat_cuda.cu
        src/operator/nn/resize_cuda.cu
        src/operator/nn/transpose_cuda.cu
        src/operator/nn/sub_cuda.cu
        src/operator/nn/div_cuda.cu
        src/operator/nn/slice_cuda.cu
        src/operator/nn/split_cuda.cu
        src/operator/nn/layernorm_cuda.cu
        src/operator/nn/gather_cuda.cu
        src/operator/nn/squeeze_unsqueeze_cuda.cu
        src/operator/nn/reduce_cuda.cu
        src/operator/nn/cast_cuda.cu
        src/operator/nn/argmax_cuda.cu
        src/operator/nn/mha_fused_cuda.cu
        src/operator/nn/mha_fused_f16_cuda.cu
        src/operator/nn/elementwise_f16_cuda.cu
        src/operator/nn/conv_f16_cuda.cu
        src/operator/nn/matmul_f16_cuda.cu
        src/operator/nn/causal_mask_cuda.cu
        src/operator/nn/mha_decode_cuda.cu
        src/operator/nn/rope_cuda.cu
        src/operator/blas/matmul_cuda.cu
    )
endif()

target_include_directories(operator PUBLIC
    src/operator/include
    src/platform/include
)

target_include_directories(operator PRIVATE
    src/operator/blas
    src/operator/nn
)

target_link_libraries(operator PUBLIC platform)

# ============================================================
# Application layer (static library / executable)
# ============================================================
add_library(application STATIC
    src/application/engine/graph.c
)

target_include_directories(application PUBLIC
    src/application/include
    src/application/engine
    src/operator/include
    src/platform/include
)

target_include_directories(application PRIVATE
    src/operator/nn      # for conv_int.h (kernel fusion)
)

target_link_libraries(application PUBLIC operator platform)

# ============================================================
# ONNX model loading library
# ============================================================
add_library(onnx_model STATIC
    src/application/model/onnx_parser.c
    src/application/model/onnx_loader.c
    src/application/engine/inference_engine.c
)

target_include_directories(onnx_model PUBLIC
    src/application/model
    src/application/engine
    src/application/include
    src/operator/include
    src/operator/blas
    src/operator/nn
    src/platform/include
)

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(onnx_model PRIVATE /wd4189 /wd4456 /wd4101)
endif()

target_link_libraries(onnx_model PUBLIC application operator platform)

# ============================================================
# Main executable
# ============================================================
add_executable(cudaforge src/main.c)
target_link_libraries(cudaforge onnx_model application operator platform)

# ============================================================
# Tests
# ============================================================
if(ENABLE_TESTS)
    enable_testing()

    # Unity test framework (local copy, no git fetch needed)
    add_subdirectory(third_party/unity)
    include_directories(${unity_SOURCE_DIR}/src)
    target_compile_options(unity PRIVATE /WX-)

    add_executable(test_relu
        src/operator/nn/test/test_relu.c
    )
    target_include_directories(test_relu PRIVATE third_party/unity/src)
    target_link_libraries(test_relu PRIVATE operator platform unity)

    add_test(NAME test_relu COMMAND test_relu)

    add_executable(test_matmul
        src/operator/blas/test/test_matmul.c
    )
    target_include_directories(test_matmul PRIVATE
        third_party/unity/src
        src/operator/blas       # for matmul_int.h
    )
    target_link_libraries(test_matmul PRIVATE operator platform unity)
    add_test(NAME test_matmul COMMAND test_matmul)

    add_executable(test_conv
        src/operator/nn/test/test_conv.c
    )
    target_include_directories(test_conv PRIVATE
        third_party/unity/src
        src/operator/nn
        src/operator/blas
    )
    target_link_libraries(test_conv PRIVATE operator platform unity)
    add_test(NAME test_conv COMMAND test_conv)

    if(ENABLE_CUDA)
    add_executable(test_conv_scale tests/test_conv_scale.c)
    target_include_directories(test_conv_scale PRIVATE
        third_party/unity/src
        src/operator/nn
        src/operator/blas
        src/operator/include
        src/platform/include
        src/platform/cuda
    )
    target_link_libraries(test_conv_scale PRIVATE operator platform)
    target_compile_definitions(test_conv_scale PRIVATE USE_CUDA)
    add_test(NAME test_conv_scale COMMAND test_conv_scale)

    add_executable(test_ops_scale tests/test_ops_scale.c)
    target_include_directories(test_ops_scale PRIVATE
        third_party/unity/src
        src/operator/nn
        src/operator/blas
        src/operator/include
        src/platform/include
        src/platform/cuda
    )
    target_link_libraries(test_ops_scale PRIVATE operator platform)
    target_compile_definitions(test_ops_scale PRIVATE USE_CUDA)
    add_test(NAME test_ops_scale COMMAND test_ops_scale)
    endif()

    add_executable(test_activations
        src/operator/nn/test/test_activations.c
    )
    target_include_directories(test_activations PRIVATE third_party/unity/src)
    target_link_libraries(test_activations PRIVATE operator platform unity)
    add_test(NAME test_activations COMMAND test_activations)

    add_executable(test_pooling
        src/operator/nn/test/test_pooling.c
    )
    target_include_directories(test_pooling PRIVATE third_party/unity/src)
    target_link_libraries(test_pooling PRIVATE operator platform unity)
    add_test(NAME test_pooling COMMAND test_pooling)

    add_executable(test_batchnorm
        src/operator/nn/test/test_batchnorm.c
    )
    target_include_directories(test_batchnorm PRIVATE third_party/unity/src)
    target_link_libraries(test_batchnorm PRIVATE operator platform unity)
    add_test(NAME test_batchnorm COMMAND test_batchnorm)

    add_executable(test_add
        src/operator/nn/test/test_add.c
    )
    target_include_directories(test_add PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_add PRIVATE operator platform unity)
    add_test(NAME test_add COMMAND test_add)

    add_executable(test_reshape
        src/operator/nn/test/test_reshape.c
    )
    target_include_directories(test_reshape PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_reshape PRIVATE operator platform unity)
    add_test(NAME test_reshape COMMAND test_reshape)

    add_executable(test_globalavgpool
        src/operator/nn/test/test_globalavgpool.c
    )
    target_include_directories(test_globalavgpool PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_globalavgpool PRIVATE operator platform unity)
    add_test(NAME test_globalavgpool COMMAND test_globalavgpool)

    add_executable(test_softmax
        src/operator/nn/test/test_softmax.c
    )
    target_include_directories(test_softmax PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_softmax PRIVATE operator platform unity)
    add_test(NAME test_softmax COMMAND test_softmax)

    add_executable(test_mul
        src/operator/nn/test/test_mul.c
    )
    target_include_directories(test_mul PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_mul PRIVATE operator platform unity)
    add_test(NAME test_mul COMMAND test_mul)

    add_executable(test_concat
        src/operator/nn/test/test_concat.c
    )
    target_include_directories(test_concat PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_concat PRIVATE operator platform unity)
    add_test(NAME test_concat COMMAND test_concat)

    add_executable(test_resize
        src/operator/nn/test/test_resize.c
    )
    target_include_directories(test_resize PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_resize PRIVATE operator platform unity)
    add_test(NAME test_resize COMMAND test_resize)

    add_executable(test_transpose
        src/operator/nn/test/test_transpose.c
    )
    target_include_directories(test_transpose PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_transpose PRIVATE operator platform unity)
    add_test(NAME test_transpose COMMAND test_transpose)

    add_executable(test_sub
        src/operator/nn/test/test_sub.c
    )
    target_include_directories(test_sub PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_sub PRIVATE operator platform unity)
    add_test(NAME test_sub COMMAND test_sub)

    add_executable(test_div
        src/operator/nn/test/test_div.c
    )
    target_include_directories(test_div PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_div PRIVATE operator platform unity)
    add_test(NAME test_div COMMAND test_div)

    add_executable(test_slice
        src/operator/nn/test/test_slice.c
    )
    target_include_directories(test_slice PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_slice PRIVATE operator platform unity)
    add_test(NAME test_slice COMMAND test_slice)

    add_executable(test_split
        src/operator/nn/test/test_split.c
    )
    target_include_directories(test_split PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_split PRIVATE operator platform unity)
    add_test(NAME test_split COMMAND test_split)

    add_executable(test_layernorm
        src/operator/nn/test/test_layernorm.c
    )
    target_include_directories(test_layernorm PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_layernorm PRIVATE operator platform unity)
    add_test(NAME test_layernorm COMMAND test_layernorm)

    add_executable(test_gather
        src/operator/nn/test/test_gather.c
    )
    target_include_directories(test_gather PRIVATE
        third_party/unity/src
        src/operator/nn
    )
    target_link_libraries(test_gather PRIVATE operator platform unity)
    add_test(NAME test_gather COMMAND test_gather)

    add_executable(test_graph
        src/application/engine/test/test_graph.c
    )
    target_include_directories(test_graph PRIVATE
        third_party/unity/src
        src/operator/nn       # for conv_int.h
    )
    target_link_libraries(test_graph PRIVATE application operator platform unity)
    add_test(NAME test_graph COMMAND test_graph)

    if(ENABLE_CUDA)
        add_executable(test_cuda_kernels
            tests/test_cuda_kernels.cu
        )
        target_include_directories(test_cuda_kernels PRIVATE
            third_party/unity/src
            src/operator/blas
            src/operator/nn
        )
        target_link_libraries(test_cuda_kernels PRIVATE operator platform unity)
        add_test(NAME test_cuda_kernels COMMAND test_cuda_kernels)

        # Benchmarks (not added as ctest tests)
        add_executable(bench_operators
            tests/bench_operators.cu
        )
        target_include_directories(bench_operators PRIVATE
            third_party/unity/src
            src/operator/blas
            src/operator/nn
        )
        target_link_libraries(bench_operators PRIVATE operator platform unity)
    endif()

    add_executable(test_onnx
        tests/test_onnx.c
    )
    target_include_directories(test_onnx PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_onnx PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_onnx PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_onnx COMMAND test_onnx)

    add_executable(test_classification
        tests/test_classification.c
    )
    target_include_directories(test_classification PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_classification PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_classification PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_classification COMMAND test_classification)
    set_tests_properties(test_classification PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(test_resnet
        tests/test_resnet.c
    )
    target_include_directories(test_resnet PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_resnet PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_resnet PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_resnet COMMAND test_resnet)
    set_tests_properties(test_resnet PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(test_yolo
        tests/test_yolo.c
    )
    target_include_directories(test_yolo PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_yolo PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_yolo PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_yolo COMMAND test_yolo)
    set_tests_properties(test_yolo PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(test_reduce
        tests/test_reduce.c
    )
    target_include_directories(test_reduce PRIVATE
        third_party/unity/src
        src/operator/include
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_reduce PRIVATE operator platform unity)
    target_compile_definitions(test_reduce PRIVATE USE_CUDA)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_reduce PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_reduce COMMAND test_reduce)

    add_executable(test_bert
        tests/test_bert.c
    )
    target_include_directories(test_bert PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_bert PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_bert PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_bert COMMAND test_bert)
    set_tests_properties(test_bert PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(test_bert_mha
        tests/test_bert_mha.c
    )
    target_include_directories(test_bert_mha PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_bert_mha PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_bert_mha PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_bert_mha COMMAND test_bert_mha)
    set_tests_properties(test_bert_mha PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(test_mha_decode
        tests/test_mha_decode.c
    )
    target_include_directories(test_mha_decode PRIVATE
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(test_mha_decode PRIVATE onnx_model application operator platform)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(test_mha_decode PRIVATE /wd4189 /wd4456 /wd4101)
    endif()
    add_test(NAME test_mha_decode COMMAND test_mha_decode)
    set_tests_properties(test_mha_decode PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    add_executable(bench_bert_mha
        tests/bench_bert_mha.c
    )
    target_include_directories(bench_bert_mha PRIVATE
        third_party/unity/src
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(bench_bert_mha PRIVATE onnx_model application operator platform unity)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(bench_bert_mha PRIVATE /wd4189 /wd4456 /wd4101)
    endif()

    add_executable(bench_mha_decode
        tests/bench_mha_decode.c
    )
    target_include_directories(bench_mha_decode PRIVATE
        src/application/model
        src/application/engine
        src/application/include
        src/operator/include
        src/operator/blas
        src/operator/nn
        src/platform/include
    )
    target_link_libraries(bench_mha_decode PRIVATE onnx_model application operator platform)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(bench_mha_decode PRIVATE /wd4189 /wd4456 /wd4101)
    endif()

endif()

# ============================================================
# Coverage
# ============================================================
if(ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()
