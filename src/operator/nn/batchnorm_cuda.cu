#include "operator.h"
#include "cuda_ops.h"
#include "batchnorm_int.h"
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cooperative_groups.h>

/* ============================================================
 * Inference kernel — coalesced memory access.
 * 2D block: x-dim handles contiguous spatial elements,
 *           y-dim handles different channels.
 * Grid: x covers spatial blocks, y covers channel blocks.
 * This replaces the original strided-access kernel.
 * ============================================================ */
#define BN_BLOCK_X 256
#define BN_BLOCK_Y 1

__global__ void batchnorm_f32_inference_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ y,
    int64_t C, int64_t hw, float epsilon)
{
    int c = blockIdx.y;
    int spatial = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || spatial >= hw) return;

    float scale = gamma[c] / sqrtf(var[c] + epsilon);
    float bias  = beta[c] - scale * mean[c];

    int64_t idx = c * hw + spatial;
    y[idx] = x[idx] * scale + bias;
}

/* ============================================================
 * Training kernel — single-pass with cross-block reduction.
 * Phase 1: each block computes partial sum/sum_sq for its
 *          chunk of data within a channel.
 * Phase 2: block 0 reduces partials, computes global stats,
 *          and applies normalization.
 * Uses cooperative_groups grid sync (sm_60+).
 * ============================================================ */
__global__ void batchnorm_f32_train_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int64_t C, int64_t hw, float epsilon, float momentum)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    /* Shared memory for per-block partial sums */
    __shared__ float block_sum[BN_BLOCK_X];
    __shared__ float block_sum_sq[BN_BLOCK_X];

    int c     = blockIdx.y;
    int tid   = threadIdx.x;
    int spatial = blockIdx.x * blockDim.x + tid;

    float local_sum   = 0.0f;
    float local_sum_sq = 0.0f;

    if (c < C && spatial < hw) {
        int64_t idx = c * hw + spatial;
        float val = x[idx];
        local_sum   = val;
        local_sum_sq = val * val;
    }

    /* Warp-level reduction within block */
    block_sum[tid]    = local_sum;
    block_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid]    += block_sum[tid + s];
            block_sum_sq[tid] += block_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float partial_sum    = block_sum[0];
    float partial_sum_sq = block_sum_sq[0];

    /* Cross-block reduction via global memory + atomics */
    if (tid == 0 && c < C) {
        atomicAdd(&running_mean[c], partial_sum);
        atomicAdd(&running_var[c], partial_sum_sq);
    }

    /* Grid sync: all blocks must finish accumulating before normalization */
    grid.sync();

    /* After sync, running_mean[c] = sum_x, running_var[c] = sum_x² */
    if (c < C && spatial < hw) {
        float global_mean = running_mean[c] / (float)hw;
        float global_var  = running_var[c] / (float)hw - global_mean * global_mean;
        global_var = fmaxf(global_var, 0.0f);

        float scale = gamma[c] / sqrtf(global_var + epsilon);
        float bias  = beta[c] - scale * global_mean;

        int64_t idx = c * hw + spatial;
        y[idx] = x[idx] * scale + bias;

        /* Update running stats if first spatial element */
        if (spatial == 0) {
            running_mean[c] = momentum * global_mean + (1.0f - momentum) * running_mean[c];
            running_var[c]  = momentum * global_var  + (1.0f - momentum) * running_var[c];
        }
    }
}

/* ============================================================
 * Helper: launch batchnorm_train with cooperative kernel support
 * ============================================================ */
static int batchnorm_f32_train_launch(
    const float* x, const float* gamma, const float* beta,
    float* y, float* running_mean, float* running_var,
    int64_t C, int64_t hw, float epsilon, float momentum,
    cudaStream_t s)
{
    int spatial_blocks = (int)((hw + BN_BLOCK_X - 1) / BN_BLOCK_X);
    dim3 block(BN_BLOCK_X, 1, 1);
    dim3 grid(spatial_blocks, C, 1);

    size_t smem = BN_BLOCK_X * sizeof(float) * 2;

    void* args[] = {
        (void*)&x, (void*)&gamma, (void*)&beta,
        (void*)&y, (void*)&running_mean, (void*)&running_var,
        (void*)&C, (void*)&hw, (void*)&epsilon, (void*)&momentum
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)batchnorm_f32_train_kernel,
        grid, block, args, smem, s);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

/* ============================================================
 * Dispatch
 * ============================================================ */
int batchnorm_f32_cuda(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] ||
        !inputs[3] || !inputs[4] || !inputs[5] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const batchnorm_params_t* p = (const batchnorm_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t hw = *(const int64_t*)inputs[5];

    int spatial_blocks = (int)((hw + BN_BLOCK_X - 1) / BN_BLOCK_X);
    dim3 block(BN_BLOCK_X, 1, 1);
    dim3 grid(spatial_blocks, p->C, 1);

    return CUDA_KERNEL_LAUNCH(batchnorm_f32_inference_kernel, grid, block, 0, s,
                       (const float*)inputs[0],
                       (const float*)inputs[1],
                       (const float*)inputs[2],
                       (const float*)inputs[3],
                       (const float*)inputs[4],
                       (float*)outputs[0],
                       p->C, hw, p->epsilon);
}

extern "C" int register_batchnorm_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "batchnorm_f32_cuda", .data_type = "f32",
        .func = batchnorm_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
