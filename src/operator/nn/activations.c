#include "operator.h"
#include <math.h>

/* Sigmoid: 1 / (1 + exp(-x)) */
int sigmoid_f32(const void* inputs[], void* outputs[],
                const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
    return 0;
}

/* GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

int gelu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        out[i] = gelu_tanh(in[i]);
    }
    return 0;
}

/* SiLU (Sigmoid Linear Unit): x * sigmoid(x) */
int silu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
    return 0;
}

static const operator_registry_t s_sigmoid_reg = {
    .name = "sigmoid_f32", .data_type = "f32",
    .func = sigmoid_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

static const operator_registry_t s_gelu_reg = {
    .name = "gelu_f32", .data_type = "f32",
    .func = gelu_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

/* Exp: y = exp(x) */
int exp_f32(const void* inputs[], void* outputs[],
            const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }
    return 0;
}

static const operator_registry_t s_silu_reg = {
    .name = "silu_f32", .data_type = "f32",
    .func = silu_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

static const operator_registry_t s_exp_reg = {
    .name = "exp_f32", .data_type = "f32",
    .func = exp_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

/* Tanh */
int tanh_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];
    for (int64_t i = 0; i < n; i++) out[i] = tanhf(in[i]);
    return 0;
}

static const operator_registry_t s_tanh_reg = {
    .name = "tanh_f32", .data_type = "f32",
    .func = tanh_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

int register_activations(void) {
    int ret = 0;
    ret += operator_register(&s_sigmoid_reg);
    ret += operator_register(&s_gelu_reg);
    ret += operator_register(&s_silu_reg);
    ret += operator_register(&s_exp_reg);
    ret += operator_register(&s_tanh_reg);
    return ret;
}
