#include "operator.h"
#include "cuda_ops.h"
#include "layernorm_int.h"

/* Block-level parallel reduction for mean and variance.
 * Each block handles one normalization instance.
 * Block size must be a power of 2 <= OPS_THREADS_PER_BLOCK. */

#define LN_WARP_SIZE 32

/* Compute mean via shared-memory reduction, then normalize in-place */
__global__ void layernorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int64_t N, int64_t D, float epsilon)
{
    extern __shared__ float s_buf[];
    float* s_data = s_buf;                    /* D elements (if D <= blockDim.x) */

    int64_t n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    int D_int = (int)D;

    const float* xn = x + n * D;
    float*       yn = y + n * D;
    const float* gn = gamma;
    const float* bn = beta;

    /* Load elements into shared memory, compute local sum for mean */
    float local_sum = 0.0f;
    for (int i = tid; i < D_int; i += blockDim.x) {
        float val = xn[i];
        s_data[i] = val;
        local_sum += val;
    }

    /* Shared-memory parallel reduction for mean */
    s_data[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }

    float mean = s_data[0] / (float)D;

    /* Compute variance: local sum of squared diffs */
    float local_var = 0.0f;
    for (int i = tid; i < D_int; i += blockDim.x) {
        float diff = xn[i] - mean;
        local_var += diff * diff;
    }

    s_data[tid] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }

    float inv_std = rsqrtf(s_data[0] / (float)D + epsilon);

    /* Normalize and write */
    for (int i = tid; i < D_int; i += blockDim.x) {
        float val = xn[i];
        yn[i] = (val - mean) * inv_std * gn[i] + bn[i];
    }
}

/* ============================================================
 * Fused LayerNorm + Residual Add
 * y = LayerNorm(x + residual) * gamma + beta
 * Saves one full read+write pass over the residual tensor.
 * ============================================================ */
__global__ void layernorm_residual_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int64_t N, int64_t D, float epsilon)
{
    extern __shared__ float s_buf[];
    float* s_data = s_buf;

    int64_t n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    int D_int = (int)D;

    const float* xn = x + n * D;
    const float* rn = residual + n * D;
    float*       yn = y + n * D;
    const float* gn = gamma;
    const float* bn = beta;

    /* Compute mean of (x + residual) */
    float local_sum = 0.0f;
    for (int i = tid; i < D_int; i += blockDim.x)
        local_sum += xn[i] + rn[i];

    s_data[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    float mean = s_data[0] / (float)D;

    /* Compute variance */
    float local_var = 0.0f;
    for (int i = tid; i < D_int; i += blockDim.x) {
        float diff = xn[i] + rn[i] - mean;
        local_var += diff * diff;
    }

    s_data[tid] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(s_data[0] / (float)D + epsilon);

    /* Normalize with fused residual */
    for (int i = tid; i < D_int; i += blockDim.x) {
        float val = xn[i] + rn[i];
        yn[i] = (val - mean) * inv_std * gn[i] + bn[i];
    }
}

int layernorm_f32_cuda(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const layernorm_params_t* p = (const layernorm_params_t*)params;
    const float* x     = (const float*)inputs[0];
    const float* gamma = (const float*)inputs[1];
    const float* beta  = (const float*)inputs[2];
    float* y           = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int D_int = (int)p->normalized_size;
    int block_sz = D_int < OPS_THREADS_PER_BLOCK ? D_int : OPS_THREADS_PER_BLOCK;
    /* Round down to nearest power of 2 for clean reduction */
    int pow2 = 1;
    while (pow2 * 2 <= block_sz) pow2 <<= 1;
    block_sz = pow2;

    size_t smem = (size_t)D_int * sizeof(float);
    dim3 grid((unsigned int)p->N, 1, 1);
    dim3 block((unsigned int)block_sz, 1, 1);

    return CUDA_KERNEL_LAUNCH(layernorm_f32_kernel, grid, block, smem, s,
                               x, gamma, beta, y, p->N, p->normalized_size, p->epsilon);
}

extern "C" int register_layernorm_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "layernorm_f32_cuda", .data_type = "f32",
        .func = layernorm_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
