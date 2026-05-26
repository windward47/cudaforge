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
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    matmul_params_t mp = {.M = K, .N = col_cols, .K = col_rows};
    const void* mat_inputs[]  = {w, col_buf, NULL};
    void*       mat_outputs[] = {NULL};

    for (int64_t n = 0; n < N; n++) {
        const float* in_n  = in  + n * C * H * W;
        float*       out_n = out + n * K * OH * OW;

        int im2col_ret = CUDA_KERNEL_LAUNCH(im2col_f32_kernel, grid, block, 0, s,
                           in_n, col_buf, C, H, W, KH, KW,
                           p->pad_h, p->pad_w,
                           p->stride_h, p->stride_w,
                           p->dilation_h, p->dilation_w, OH, OW);
        if (im2col_ret != 0) { g_cuda.device_free(col_buf); return im2col_ret; }

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
    } else if (fuse_activation == 13) {
        sum = sum / (1.0f + expf(-sum));                       /* SiLU */
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
        } else if (fuse_activation == 13) {
            sum = sum / (1.0f + expf(-sum));                       /* SiLU */
        }
        output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
    }
}

/* ============================================================
 * Winograd F(2,3) — 2×2 output tile from 3×3 filter via 4×4 transform.
 * References: Lavin & Gray (2016), cuDNN Winograd convolution.
 * ============================================================ */

/* Pre-transform 3×3 weight to 4×4 Winograd domain: V = G * g * G^T
 * G is 4×3, g is 3×3, V is 4×4.  One thread per (k,c) pair. */
__global__ void winograd_f23_weight_transform_kernel(
    const float* __restrict__ weight,  /* K × C × 3 × 3 */
    float* __restrict__ V,             /* K × C × 4 × 4 */
    int64_t K, int64_t C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = K * C;
    if (idx >= total) return;

    int k = (int)(idx / C);
    int c = (int)(idx % C);

    const float* g = weight + (k * C + c) * 9;
    float* v       = V      + (k * C + c) * 16;

    /* Load 3×3 filter */
    float g00 = g[0], g01 = g[1], g02 = g[2];
    float g10 = g[3], g11 = g[4], g12 = g[5];
    float g20 = g[6], g21 = g[7], g22 = g[8];

    /* G * g (4×3 * 3×3 = 4×3):
     * G = [[1,    0,    0],
     *      [1/2,  1/2,  1/2],
     *      [1/2, -1/2,  1/2],
     *      [0,    0,    1]]  */
    float Gg[12];
    Gg[0] = g00;   Gg[1] = g01;   Gg[2] = g02;
    Gg[3] = 0.5f * (g00 + g10 + g20);
    Gg[4] = 0.5f * (g01 + g11 + g21);
    Gg[5] = 0.5f * (g02 + g12 + g22);
    Gg[6] = 0.5f * (g00 - g10 + g20);
    Gg[7] = 0.5f * (g01 - g11 + g21);
    Gg[8] = 0.5f * (g02 - g12 + g22);
    Gg[9] = g20;  Gg[10] = g21;  Gg[11] = g22;

    /* (G*g) * G^T  (4×3 * 3×4 = 4×4)
     * G^T = [[1, 1/2, 1/2, 0],
     *        [0, 1/2,-1/2, 0],
     *        [0, 1/2, 1/2, 1]] */
    v[0]  = Gg[0];
    v[1]  = 0.5f * (Gg[0] + Gg[1] + Gg[2]);
    v[2]  = 0.5f * (Gg[0] - Gg[1] + Gg[2]);
    v[3]  = Gg[2];

    v[4]  = Gg[3];
    v[5]  = 0.5f * (Gg[3] + Gg[4] + Gg[5]);
    v[6]  = 0.5f * (Gg[3] - Gg[4] + Gg[5]);
    v[7]  = Gg[5];

    v[8]  = Gg[6];
    v[9]  = 0.5f * (Gg[6] + Gg[7] + Gg[8]);
    v[10] = 0.5f * (Gg[6] - Gg[7] + Gg[8]);
    v[11] = Gg[8];

    v[12] = Gg[9];
    v[13] = 0.5f * (Gg[9] + Gg[10] + Gg[11]);
    v[14] = 0.5f * (Gg[9] - Gg[10] + Gg[11]);
    v[15] = Gg[11];
}

/* Winograd F(2,3) convolution kernel.
 * Each thread computes a 2×2 output tile for one (n,k) pair. */
__global__ void winograd_f23_kernel(
    const float* __restrict__ input,
    const float* __restrict__ V,        /* pre-transformed weights */
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t K,
    int64_t OH, int64_t OW,
    int64_t pad_h, int64_t pad_w,
    int64_t fuse_activation)
{
    int tiles_h = (int)((OH + 1) / 2);  /* ceil(OH/2) */
    int tiles_w = (int)((OW + 1) / 2);
    int tiles_per_nk = tiles_h * tiles_w;

    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = N * K * tiles_per_nk;
    if (idx >= total) return;

    int nk       = (int)(idx / tiles_per_nk);
    int tile_idx = (int)(idx % tiles_per_nk);
    int tile_h   = tile_idx / tiles_w;
    int tile_w   = tile_idx % tiles_w;
    int n = nk / (int)K;
    int k = nk % (int)K;

    int oh0 = tile_h * 2;
    int ow0 = tile_w * 2;
    int ih0 = oh0 - (int)pad_h;
    int iw0 = ow0 - (int)pad_w;

    /* Accumulate over channels in Winograd domain */
    float M[16] = {0};

    const float* in_n = input + n * C * H * W;
    const float* V_k  = V     + k * C * 16;

    for (int64_t c = 0; c < C; c++) {
        /* Load 4×4 input tile d */
        float d[16];
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                int ih = ih0 + y;
                int iw = iw0 + x;
                d[y * 4 + x] = (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                    ? in_n[c * H * W + ih * W + iw] : 0.0f;
            }
        }

        /* Input transform: U = B^T * d * B
         * B^T = [[1, 0,-1, 0],
         *        [0, 1, 1, 0],
         *        [0,-1, 1, 0],
         *        [0,-1, 0, 1]]  */

        /* Step 1: T = d * B (4×4) */
        float T[16];
#define T_(y,x) T[(y)*4+(x)]
#define d_(y,x) d[(y)*4+(x)]
        for (int y = 0; y < 4; y++) {
            T_(y,0) = d_(y,0) - d_(y,2);
            T_(y,1) = d_(y,1) + d_(y,2);
            T_(y,2) = -d_(y,1) + d_(y,2);
            T_(y,3) = d_(y,1) - d_(y,3);
        }
#undef d_

        /* Step 2: U = B^T * T (4×4) */
        float U[16];
#define U_(y,x) U[(y)*4+(x)]
        for (int x = 0; x < 4; x++) {
            U_(0,x) = T_(0,x) - T_(2,x);
            U_(1,x) = T_(1,x) + T_(2,x);
            U_(2,x) = -T_(1,x) + T_(2,x);
            U_(3,x) = T_(1,x) - T_(3,x);
        }
#undef T_
#undef U_

        /* Element-wise multiply and accumulate:
         * M += U ⊙ V_kc */
        const float* V_kc = V_k + c * 16;
        for (int i = 0; i < 16; i++) {
            M[i] += U[i] * V_kc[i];
        }
    }

    /* Output inverse transform: Y = A^T * M * A
     * A^T = [[1, 1, 1, 0],
     *        [0, 1,-1,-1]]  */

    /* Step 1: T_out = A^T * M (2×4) */
    float T_y[8];
#define M_(y,x)  M[(y)*4+(x)]
#define TY_(y,x) T_y[(y)*4+(x)]
    for (int x = 0; x < 4; x++) {
        TY_(0,x) = M_(0,x) + M_(1,x) + M_(2,x);
        TY_(1,x) = M_(1,x) - M_(2,x) - M_(3,x);
    }
#undef M_

    /* Step 2: Y = T_out * A (2×4 * 4×2 = 2×2) */
    float o00 = TY_(0,0) + TY_(0,1) + TY_(0,2);
    float o01 = TY_(0,1) - TY_(0,2) - TY_(0,3);
    float o10 = TY_(1,0) + TY_(1,1) + TY_(1,2);
    float o11 = TY_(1,1) - TY_(1,2) - TY_(1,3);
#undef TY_

    /* Bias and activation */
    if (bias) {
        float bv = bias[k];
        o00 += bv; o01 += bv; o10 += bv; o11 += bv;
    }

    if (fuse_activation == 1) {
        o00 = o00 > 0.0f ? o00 : 0.0f;
        o01 = o01 > 0.0f ? o01 : 0.0f;
        o10 = o10 > 0.0f ? o10 : 0.0f;
        o11 = o11 > 0.0f ? o11 : 0.0f;
    } else if (fuse_activation == 2) {
        o00 = 1.0f / (1.0f + expf(-o00));
        o01 = 1.0f / (1.0f + expf(-o01));
        o10 = 1.0f / (1.0f + expf(-o10));
        o11 = 1.0f / (1.0f + expf(-o11));
    } else if (fuse_activation == 3) {
        float t0 = tanhf(0.79788456f * (o00 + 0.044715f * o00 * o00 * o00));
        o00 = 0.5f * o00 * (1.0f + t0);
        float t1 = tanhf(0.79788456f * (o01 + 0.044715f * o01 * o01 * o01));
        o01 = 0.5f * o01 * (1.0f + t1);
        float t2 = tanhf(0.79788456f * (o10 + 0.044715f * o10 * o10 * o10));
        o10 = 0.5f * o10 * (1.0f + t2);
        float t3 = tanhf(0.79788456f * (o11 + 0.044715f * o11 * o11 * o11));
        o11 = 0.5f * o11 * (1.0f + t3);
    } else if (fuse_activation == 13) {
        o00 = o00 / (1.0f + expf(-o00));
        o01 = o01 / (1.0f + expf(-o01));
        o10 = o10 / (1.0f + expf(-o10));
        o11 = o11 / (1.0f + expf(-o11));
    }

    /* Write output — skip positions beyond OH/OW boundary */
    float* out_nk = output + n * K * OH * OW + k * OH * OW;
    if (oh0 < (int)OH && ow0 < (int)OW) out_nk[oh0     * OW + ow0]     = o00;
    if (oh0 < (int)OH && ow0+1 < (int)OW) out_nk[oh0     * OW + ow0 + 1] = o01;
    if (oh0+1 < (int)OH && ow0 < (int)OW) out_nk[(oh0+1) * OW + ow0]     = o10;
    if (oh0+1 < (int)OH && ow0+1 < (int)OW) out_nk[(oh0+1) * OW + ow0 + 1] = o11;
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

    /* Winograd F(2,3) path: 3×3 kernel, stride=1, dilation=1, no groups */
    if (KH == 3 && KW == 3 && p->stride_h == 1 && p->stride_w == 1
        && p->dilation_h == 1 && p->dilation_w == 1 && p->groups == 1) {
        int64_t V_size = K * C * 16;
        float* V = (float*)g_cuda.device_alloc((size_t)V_size * sizeof(float));
        if (!V) return -1;

        int wgrid = (int)((K * C + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK);
        int wblock = OPS_THREADS_PER_BLOCK;
        int ret = CUDA_KERNEL_LAUNCH(winograd_f23_weight_transform_kernel,
                                      wgrid, wblock, 0, s, w, V, K, C);
        if (ret != 0) { g_cuda.device_free(V); return ret; }

        int tiles_h = (int)((OH + 1) / 2);
        int tiles_w = (int)((OW + 1) / 2);
        int64_t total_tiles = N * K * tiles_h * tiles_w;
        int tile_grid = (int)((total_tiles + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK);
        int tile_block = OPS_THREADS_PER_BLOCK;

        ret = CUDA_KERNEL_LAUNCH(winograd_f23_kernel, tile_grid, tile_block, 0, s,
                                  in, V,
                                  p->fuse_activation ? bias : NULL,
                                  out, N, C, H, W, K, OH, OW,
                                  p->pad_h, p->pad_w,
                                  p->fuse_activation);
        g_cuda.device_free(V);
        if (ret != 0) return ret;

        /* Separate bias add if not fused */
        if (bias && p->fuse_activation == 0) {
            int64_t total = N * K * OH * OW;
            int bg = (int)((total + 256 - 1) / 256);
            return CUDA_KERNEL_LAUNCH(conv_bias_add_kernel, bg, 256, 0, s,
                                       out, bias, N, K, OH, OW);
        }
        return 0;
    }

    dim3 block(CONV_DIRECT_TILE_W, CONV_DIRECT_TILE_H, 1);
    dim3 grid((unsigned int)((OW + CONV_DIRECT_TILE_W - 1) / CONV_DIRECT_TILE_W),
              (unsigned int)((OH + CONV_DIRECT_TILE_H - 1) / CONV_DIRECT_TILE_H),
              (unsigned int)(N * K));

    /* When fusing activation, pass bias to kernel for inline bias+activation.
     * Otherwise, leave bias for the separate bias-add kernel below. */
    int ret = CUDA_KERNEL_LAUNCH(conv2d_f32_direct_kernel, grid, block, 0, s,
                       in, w,
                       p->fuse_activation ? bias : NULL,
                       out, C, H, W, K, KH, KW,
                       OH, OW,
                       p->stride_h, p->stride_w,
                       p->pad_h, p->pad_w,
                       p->dilation_h, p->dilation_w,
                       p->fuse_activation);
    if (ret != 0) return ret;

    /* Add bias separately only when NOT already fused into conv kernel */
    if (bias && p->fuse_activation == 0) {
        int64_t total = N * K * OH * OW;
        int block_sz = 256;
        int grid_sz = (int)((total + block_sz - 1) / block_sz);
        return CUDA_KERNEL_LAUNCH(conv_bias_add_kernel, grid_sz, block_sz, 0, s,
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
