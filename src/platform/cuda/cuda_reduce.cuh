/**
 * @file cuda_reduce.cuh
 * @brief Warp-level and block-level reduction primitives.
 *
 * Reference: flash-attention-main/csrc/flash_attn/src/softmax.h
 *
 * Provides:
 *   - warp_reduce_sum<T>()  : sum across 32 lanes via __shfl_xor_sync
 *   - warp_reduce_max<T>()  : max across 32 lanes via __shfl_xor_sync
 *   - block_reduce_sum<T>() : sum across all threads in a block
 *   - block_reduce_max<T>() : max across all threads in a block
 *
 * All functions are __device__ __forceinline__ for zero overhead.
 */
#ifndef CUDA_REDUCE_CUH_
#define CUDA_REDUCE_CUH_

#include <cuda_runtime.h>
#include <float.h>

/* --------------------------------------------------------------------------
 *  Warp-level reductions (32 lanes, no shared memory needed)
 * -------------------------------------------------------------------------- */

/**
 * @brief Sum across a warp using butterfly shuffle.
 *
 * After the call, every lane in the warp holds the total sum.
 * Requires __syncwarp() before reading if divergent lanes participated.
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

/**
 * @brief Max across a warp using butterfly shuffle.
 *
 * After the call, every lane in the warp holds the maximum.
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

/* --------------------------------------------------------------------------
 *  Block-level reductions (all threads in a block)
 *
 *  Strategy:
 *    1. Each warp does warp_reduce internally (32 lanes, register shuffle)
 *    2. Lane 0 of each warp writes partial result to shared memory
 *    3. First warp reads all partials and does final warp_reduce
 *
 *  Requires a shared memory buffer of at least (blockDim.x / 32) elements.
 *  The caller must __syncthreads() after calling if the result is shared.
 * -------------------------------------------------------------------------- */

/**
 * @brief Sum across all threads in a block.
 *
 * @param val       Each thread's value
 * @param smem      Shared memory buffer, at least (blockDim.x / 32) elements
 * @param tid       Thread index (threadIdx.x)
 * @return Sum of all threads' values (valid on all threads)
 */
template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* smem, int tid) {
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    /* Step 1: warp-level reduction */
    val = warp_reduce_sum(val);

    /* Step 2: lane 0 of each warp writes to smem */
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    /* Step 3: first warp reduces all partial sums */
    if (warp_id == 0) {
        val = (tid < num_warps) ? smem[tid] : T(0);
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    return val;
}

/**
 * @brief Max across all threads in a block.
 *
 * @param val       Each thread's value
 * @param smem      Shared memory buffer, at least (blockDim.x / 32) elements
 * @param tid       Thread index (threadIdx.x)
 * @return Maximum of all threads' values (valid on all threads)
 */
template<typename T>
__device__ __forceinline__ T block_reduce_max(T val, T* smem, int tid) {
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    /* Step 1: warp-level reduction */
    val = warp_reduce_max(val);

    /* Step 2: lane 0 of each warp writes to smem */
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    /* Step 3: first warp reduces all partial maxes */
    if (warp_id == 0) {
        val = (tid < num_warps) ? smem[tid] : T(-FLT_MAX);
        val = warp_reduce_max(val);
    }
    __syncthreads();

    return val;
}

/* --------------------------------------------------------------------------
 *  Quad-level reduction (4 lanes, used in FA2 softmax)
 *
 *  In MMA layout, 4 adjacent threads share a row of the accumulator.
 *  This is a lightweight alternative to full warp reduce when only
 *  4-way reduction is needed (e.g., online softmax row max/sum).
 * -------------------------------------------------------------------------- */

/**
 * @brief Sum across 4 adjacent lanes (quad).
 */
template<typename T>
__device__ __forceinline__ T quad_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

/**
 * @brief Max across 4 adjacent lanes (quad).
 */
template<typename T>
__device__ __forceinline__ T quad_reduce_max(T val) {
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

#endif /* CUDA_REDUCE_CUH_ */
