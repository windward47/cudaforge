#include "operator.h"
#include "cuda_ops.h"
#include "conv_int.h"
#include "matmul_int.h"

/* ============================================================
 * im2col kernel (kept as fallback)
 * ============================================================ */
__global__ void im2col_f32_kernel(const float* input, float* col_buf,
                                   int64_t C, int64_t H, int64_t W,
                                   int64_t KH, int64_t KW,
                                   int64_t pad_h, int64_t pad_w,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t dil_h, int64_t dil_w,
                                   int64_t OH, int64_t OW) {
    int64_t col_rows = C * KH * KW;
    int64_t col_cols = OH * OW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= col_rows * col_cols) return;

    int64_t ci = idx / col_cols;
    int64_t oj = idx % col_cols;

    int64_t c  = ci / (KH * KW);
    int64_t kh = (ci / KW) % KH;
    int64_t kw = ci % KW;

    int64_t oh = oj / OW;
    int64_t ow = oj % OW;

    int64_t ih = oh * stride_h + kh * dil_h - pad_h;
    int64_t iw = ow * stride_w + kw * dil_w - pad_w;

    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        col_buf[idx] = input[c * H * W + ih * W + iw];
    } else {
        col_buf[idx] = 0.0f;
    }
}

/* ============================================================
 * im2col-based conv (fallback for unusual params)
 * ============================================================ */
static int conv2d_im2col(const float* in, const float* w, float* out,
                          const conv_params_t* p, cudaStream_t s) {
    int64_t N = p->N, C = p->C, H = p->H, W = p->W, K = p->K;
    int64_t KH = p->kernel_h, KW = p->kernel_w;
    int64_t OH = (H + 2 * p->pad_h - p->dilation_h * (KH - 1) - 1) / p->stride_h + 1;
    int64_t OW = (W + 2 * p->pad_w - p->dilation_w * (KW - 1) - 1) / p->stride_w + 1;
    int64_t col_rows = C * KH * KW;
    int64_t col_cols = OH * OW;

    float* col_buf = (float*)g_cuda.device_alloc((size_t)col_rows * col_cols * sizeof(float));
    if (!col_buf) return -1;

    int64_t total = col_rows * col_cols;
    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)((total + 255) / 256), 1, 1);

    matmul_params_t mp = {.M = K, .N = col_cols, .K = col_rows};
    const void* mat_inputs[]  = {w, col_buf, NULL};
    void*       mat_outputs[] = {NULL};

    for (int64_t n = 0; n < N; n++) {
        const float* in_n  = in  + n * C * H * W;
        float*       out_n = out + n * K * OH * OW;

        CUDA_KERNEL_LAUNCH(im2col_f32_kernel, grid, block, 0, s,
                           in_n, col_buf, C, H, W, KH, KW,
                           p->pad_h, p->pad_w,
                           p->stride_h, p->stride_w,
                           p->dilation_h, p->dilation_w, OH, OW);

        mat_outputs[0] = out_n;
        const operator_registry_t* mm = operator_find("matmul_f32_cuda");
        if (mm) mm->func(mat_inputs, mat_outputs, (const operator_params_t*)&mp, NULL);
    }

    g_cuda.device_free(col_buf);
    return 0;
}

/* ============================================================
 * Direct convolution kernel — avoids im2col memory overhead.
 * Each thread block handles a TILE_H×TILE_W output tile for
 * one (n, k) pair.  Input window is cached in shared memory.
 * ============================================================ */
#define CONV_DIRECT_TILE_H 8
#define CONV_DIRECT_TILE_W 16
#define CONV_DIRECT_THREADS (CONV_DIRECT_TILE_H * CONV_DIRECT_TILE_W)
/* Max shared memory per block — queried once from the device.
 * Falls back to 48 KB (sm_86 default) if the query fails. */

__global__ void conv2d_f32_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t KH, int64_t KW,
    int64_t OH, int64_t OW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t fuse_activation)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int oh = blockIdx.y * CONV_DIRECT_TILE_H + ty;
    int ow = blockIdx.x * CONV_DIRECT_TILE_W + tx;
    int bk = blockIdx.z;
    int k  = bk % K;
    int n  = bk / K;

    if (n >= (int64_t)gridDim.z || oh >= OH || ow >= OW) return;

    float sum = 0.0f;

    const float* in_n  = input  + n * C * H * W;
    const float* w_k   = weight + k * C * KH * KW;

    for (int c = 0; c < C; c++) {
        const float* w_kc = w_k + c * KH * KW;
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh * stride_h + kh * dil_h - pad_h;
                int iw = ow * stride_w + kw * dil_w - pad_w;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += in_n[c * H * W + ih * W + iw] * w_kc[kh * KW + kw];
                }
            }
        }
    }

    /* Bias must be added BEFORE activation */
    if (bias) sum += bias[k];

    /* Inline activation (kernel fusion) */
    if (fuse_activation == 1) {
        sum = sum > 0.0f ? sum : 0.0f;                       /* ReLU */
    } else if (fuse_activation == 2) {
        sum = 1.0f / (1.0f + expf(-sum));                     /* Sigmoid */
    } else if (fuse_activation == 3) {
        float t = tanhf(0.79788456f * (sum + 0.044715f * sum * sum * sum));
        sum = 0.5f * sum * (1.0f + t);                        /* GELU */
    }

    output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
}

/* ============================================================
 * Direct convolution with shared-memory input caching.
 * For stride-1 convolutions (most common case), each block
 * loads the input window once per channel.
 * ============================================================ */
__global__ void conv2d_f32_direct_smem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t KH, int64_t KW,
    int64_t OH, int64_t OW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t fuse_activation)
{
    /* Input window dimensions for this tile */
    int win_h = (CONV_DIRECT_TILE_H - 1) * stride_h + (KH - 1) * dil_h + 1;
    int win_w = (CONV_DIRECT_TILE_W - 1) * stride_w + (KW - 1) * dil_w + 1;

    extern __shared__ float smem[];
    float* in_win = smem;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * CONV_DIRECT_TILE_W + tx;

    int oh = blockIdx.y * CONV_DIRECT_TILE_H + ty;
    int ow = blockIdx.x * CONV_DIRECT_TILE_W + tx;
    int bk = blockIdx.z;
    int k  = bk % K;
    int n  = bk / K;

    bool valid_out = (oh < OH && ow < OW && n < (int64_t)gridDim.z);

    /* Top-left input coordinate for this tile's window */
    int tile_ih0 = blockIdx.y * CONV_DIRECT_TILE_H * stride_h - pad_h;
    int tile_iw0 = blockIdx.x * CONV_DIRECT_TILE_W * stride_w - pad_w;

    float sum = 0.0f;

    const float* in_n  = input  + n * C * H * W;
    const float* w_k   = weight + k * C * KH * KW;

    for (int c = 0; c < C; c++) {
        /* Cooperatively load input window for channel c */
        int win_elems = win_h * win_w;
        for (int i = tid; i < win_elems; i += CONV_DIRECT_THREADS) {
            int wh = i / win_w;
            int ww = i % win_w;
            int ih = tile_ih0 + wh;
            int iw = tile_iw0 + ww;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                in_win[i] = in_n[c * H * W + ih * W + iw];
            else
                in_win[i] = 0.0f;
        }
        __syncthreads();

        if (valid_out) {
            const float* w_kc = w_k + c * KH * KW;
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int ih = oh * stride_h + kh * dil_h - pad_h;
                    int iw = ow * stride_w + kw * dil_w - pad_w;
                    int wh = ih - tile_ih0;
                    int ww = iw - tile_iw0;
                    if (wh >= 0 && wh < win_h && ww >= 0 && ww < win_w) {
                        sum += in_win[wh * win_w + ww] * w_kc[kh * KW + kw];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valid_out) {
        /* Bias must be added BEFORE activation */
        if (bias) sum += bias[k];

        /* Inline activation (kernel fusion) */
        if (fuse_activation == 1) {
            sum = sum > 0.0f ? sum : 0.0f;                       /* ReLU */
        } else if (fuse_activation == 2) {
            sum = 1.0f / (1.0f + expf(-sum));                     /* Sigmoid */
        } else if (fuse_activation == 3) {
            float t = tanhf(0.79788456f * (sum + 0.044715f * sum * sum * sum));
            sum = 0.5f * sum * (1.0f + t);                        /* GELU */
        }
        output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
    }
}

/* ============================================================
 * Dispatch: chooses direct kernel or im2col fallback
 * ============================================================ */
/* Bias addition kernel: out[n,k,oh,ow] += bias[k] */
__global__ void conv_bias_add_kernel(float* out, const float* bias,
                                      int64_t N, int64_t K, int64_t OH, int64_t OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * K * OH * OW;
    if (idx >= total) return;
    /* Unpack flat index: n*K*OH*OW + k*OH*OW + oh*OW + ow */
    int64_t nk = idx / (OH * OW);
    int k = (int)(nk % K);
    float bv = bias[k];
    out[idx] += bv;
}

int conv2d_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const conv_params_t* p = (const conv_params_t*)params;
    const float* in   = (const float*)inputs[0];
    const float* w    = (const float*)inputs[1];
    const float* bias = (const float*)inputs[2];
    float* out        = (float*)outputs[0];

    int64_t N = p->N, C = p->C, H = p->H, W = p->W, K = p->K;
    int64_t KH = p->kernel_h, KW = p->kernel_w;
    int64_t OH = (H + 2 * p->pad_h - p->dilation_h * (KH - 1) - 1) / p->stride_h + 1;
    int64_t OW = (W + 2 * p->pad_w - p->dilation_w * (KW - 1) - 1) / p->stride_w + 1;

    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(CONV_DIRECT_TILE_W, CONV_DIRECT_TILE_H, 1);
    dim3 grid((unsigned int)((OW + CONV_DIRECT_TILE_W - 1) / CONV_DIRECT_TILE_W),
              (unsigned int)((OH + CONV_DIRECT_TILE_H - 1) / CONV_DIRECT_TILE_H),
              (unsigned int)(N * K));

    /* When fusing activation, pass bias to kernel for inline bias+activation.
     * Otherwise, leave bias for the separate bias-add kernel below. */
    CUDA_KERNEL_LAUNCH(conv2d_f32_direct_kernel, grid, block, 0, s,
                       in, w,
                       p->fuse_activation ? bias : NULL,
                       out, C, H, W, K, KH, KW,
                       OH, OW,
                       p->stride_h, p->stride_w,
                       p->pad_h, p->pad_w,
                       p->dilation_h, p->dilation_w,
                       p->fuse_activation);

    /* Add bias separately only when NOT already fused into conv kernel */
    if (bias && p->fuse_activation == 0) {
        int64_t total = N * K * OH * OW;
        int block_sz = 256;
        int grid_sz = (int)((total + block_sz - 1) / block_sz);
        CUDA_KERNEL_LAUNCH(conv_bias_add_kernel, grid_sz, block_sz, 0, s,
                           out, bias, N, K, OH, OW);
    }
    return 0;
}

extern "C" int register_conv2d_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "conv2d_f32_cuda", .data_type = "f32",
        .func = conv2d_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
