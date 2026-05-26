extern "C" {
#include "unity.h"
#include "platform.h"
void setUp(void) {}
void tearDown(void) {}
int operator_init_all(void);
}

#include "operator.h"
#include "matmul_int.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "batchnorm_int.h"
#include "cuda_ops.h"
#include "cuda_platform.h"
#include <stdio.h>

void test_relu_f32_cuda(void) {
    const operator_registry_t* op = operator_find("relu_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t n = 8;
    float h_in[8] = {-1.0f, 0.0f, 2.5f, -3.0f, 5.0f, -0.5f, 0.0f, 100.0f};
    float h_out[8] = {0};
    float *d_in, *d_out;
    d_in = (float*)g_cuda.device_alloc(n * sizeof(float)); TEST_ASSERT_NOT_NULL(d_in);
    d_out = (float*)g_cuda.device_alloc(n * sizeof(float)); TEST_ASSERT_NOT_NULL(d_out);
    if (g_cuda.memcpy_h2d(d_in, h_in, n * sizeof(float), 0) != 0) TEST_FAIL();
    const void* inputs[]  = {d_in, &n};
    void*       outputs[] = {d_out};
    if (op->func(inputs, outputs, NULL, NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(h_out, d_out, n * sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   h_out[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   h_out[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.5f,   h_out[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   h_out[3]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f,   h_out[4]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   h_out[5]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   h_out[6]);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, h_out[7]);
    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
}

void test_matmul_f32_cuda(void) {
    const operator_registry_t* op = operator_find("matmul_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t M = 2, N = 2, K = 3;
    float h_A[6] = {1,2,3,4,5,6}, h_B[6] = {7,8,9,10,11,12}, h_C[4] = {0};
    float *d_A, *d_B, *d_C;
    d_A = (float*)g_cuda.device_alloc(M*K*sizeof(float)); TEST_ASSERT_NOT_NULL(d_A);
    d_B = (float*)g_cuda.device_alloc(K*N*sizeof(float)); TEST_ASSERT_NOT_NULL(d_B);
    d_C = (float*)g_cuda.device_alloc(M*N*sizeof(float)); TEST_ASSERT_NOT_NULL(d_C);
    if (g_cuda.memcpy_h2d(d_A, h_A, M*K*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(d_B, h_B, K*N*sizeof(float), 0) != 0) TEST_FAIL();
    matmul_params_t p = {.M = M, .N = N, .K = K};
    const void* inputs[]  = {d_A, d_B, NULL};
    void*       outputs[] = {d_C};
    if (op->func(inputs, outputs, (const operator_params_t*)&p, NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(h_C, d_C, M*N*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 58.0f,  h_C[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 64.0f,  h_C[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 139.0f, h_C[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 154.0f, h_C[3]);
    g_cuda.device_free(d_A); g_cuda.device_free(d_B); g_cuda.device_free(d_C);
}

void test_conv2d_f32_cuda(void) {
    /* First, compute reference result using CPU conv2d */
    const operator_registry_t* cpu_op = operator_find("conv2d_f32");
    TEST_ASSERT_NOT_NULL(cpu_op);

    int64_t N=1,C=1,H=3,W=3,K=1,KH=2,KW=2,OH=(H-KH)/1+1,OW=(W-KW)/1+1;
    float h_in[9]={1,2,3,4,5,6,7,8,9}, h_w[4]={1,0,0,1};
    float cpu_out[4]={0}, gpu_out[4]={0};

    conv_params_t cp={.N=N,.C=C,.H=H,.W=W,.K=K,.kernel_h=KH,.kernel_w=KW,.stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1};
    const void* cpu_inputs[]={h_in,h_w,NULL}; void* cpu_outputs[]={cpu_out};
    if (cpu_op->func(cpu_inputs,cpu_outputs,(const operator_params_t*)&cp,NULL) != 0) TEST_FAIL();
    printf("  conv CPU: [%.1f %.1f %.1f %.1f]\n", cpu_out[0],cpu_out[1],cpu_out[2],cpu_out[3]);

    /* Now compute using CUDA */
    const operator_registry_t* cuda_op = operator_find("conv2d_f32_cuda");
    TEST_ASSERT_NOT_NULL(cuda_op);

    float *d_in,*d_w,*d_out;
    d_in = (float*)g_cuda.device_alloc(C*H*W*sizeof(float)); TEST_ASSERT_NOT_NULL(d_in);
    d_w = (float*)g_cuda.device_alloc(K*C*KH*KW*sizeof(float)); TEST_ASSERT_NOT_NULL(d_w);
    d_out = (float*)g_cuda.device_alloc(K*OH*OW*sizeof(float)); TEST_ASSERT_NOT_NULL(d_out);
    if (g_cuda.memcpy_h2d(d_in, h_in, C*H*W*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(d_w, h_w, K*C*KH*KW*sizeof(float), 0) != 0) TEST_FAIL();

    const void* cuda_inputs[]={d_in,d_w,NULL}; void* cuda_outputs[]={d_out};
    if (cuda_op->func(cuda_inputs,cuda_outputs,(const operator_params_t*)&cp,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(gpu_out, d_out, K*OH*OW*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    printf("  conv CUDA: [%.1f %.1f %.1f %.1f]\n", gpu_out[0],gpu_out[1],gpu_out[2],gpu_out[3]);

    /* Compare GPU against CPU reference */
    TEST_ASSERT_FLOAT_WITHIN(0.01f,cpu_out[0],gpu_out[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,cpu_out[1],gpu_out[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,cpu_out[2],gpu_out[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,cpu_out[3],gpu_out[3]);

    g_cuda.device_free(d_in);g_cuda.device_free(d_w);g_cuda.device_free(d_out);
}

void test_maxpool2d_f32_cuda(void) {
    const operator_registry_t* op = operator_find("maxpool2d_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t N=1,C=1,H=4,W=4,KH=2,KW=2,OH=(H-KH)/2+1,OW=(W-KW)/2+1;
    float h_in[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},h_out[4]={0};
    float *d_in,*d_out;
    d_in = (float*)g_cuda.device_alloc(C*H*W*sizeof(float)); TEST_ASSERT_NOT_NULL(d_in);
    d_out = (float*)g_cuda.device_alloc(C*OH*OW*sizeof(float)); TEST_ASSERT_NOT_NULL(d_out);
    if (g_cuda.memcpy_h2d(d_in, h_in, C*H*W*sizeof(float), 0) != 0) TEST_FAIL();
    pool_params_t pp={.N=N,.C=C,.H=H,.W=W,.kernel_h=KH,.kernel_w=KW,.stride_h=2,.stride_w=2};
    const void* inputs[]={d_in}; void* outputs[]={d_out};
    if (op->func(inputs,outputs,(const operator_params_t*)&pp,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(h_out, d_out, C*OH*OW*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_EQUAL_FLOAT(6.0f,h_out[0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f,h_out[1]);
    TEST_ASSERT_EQUAL_FLOAT(14.0f,h_out[2]);
    TEST_ASSERT_EQUAL_FLOAT(16.0f,h_out[3]);
    g_cuda.device_free(d_in);g_cuda.device_free(d_out);
}

void test_avgpool2d_f32_cuda(void) {
    const operator_registry_t* op = operator_find("avgpool2d_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t N=1,C=1,H=4,W=4,KH=2,KW=2,OH=(H-KH)/2+1,OW=(W-KW)/2+1;
    float h_in[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},h_out[4]={0};
    float *d_in,*d_out;
    d_in = (float*)g_cuda.device_alloc(C*H*W*sizeof(float)); TEST_ASSERT_NOT_NULL(d_in);
    d_out = (float*)g_cuda.device_alloc(C*OH*OW*sizeof(float)); TEST_ASSERT_NOT_NULL(d_out);
    if (g_cuda.memcpy_h2d(d_in, h_in, C*H*W*sizeof(float), 0) != 0) TEST_FAIL();
    pool_params_t pp={.N=N,.C=C,.H=H,.W=W,.kernel_h=KH,.kernel_w=KW,.stride_h=2,.stride_w=2};
    const void* inputs[]={d_in}; void* outputs[]={d_out};
    if (op->func(inputs,outputs,(const operator_params_t*)&pp,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(h_out, d_out, C*OH*OW*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_FLOAT_WITHIN(0.01f,3.5f,h_out[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,5.5f,h_out[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,11.5f,h_out[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,13.5f,h_out[3]);
    g_cuda.device_free(d_in);g_cuda.device_free(d_out);
}

void test_batchnorm_f32_cuda(void) {
    const operator_registry_t* op = operator_find("batchnorm_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t C=2,hw=2;
    float x[4]={1,2,3,4},g[2]={1,1},b[2]={0,0},m[2]={0,0},v[2]={1,1},o[4]={0};
    float *dx,*dg,*db,*dm,*dv,*dd;
    dx = (float*)g_cuda.device_alloc(C*hw*sizeof(float)); TEST_ASSERT_NOT_NULL(dx);
    dg = (float*)g_cuda.device_alloc(C*sizeof(float)); TEST_ASSERT_NOT_NULL(dg);
    db = (float*)g_cuda.device_alloc(C*sizeof(float)); TEST_ASSERT_NOT_NULL(db);
    dm = (float*)g_cuda.device_alloc(C*sizeof(float)); TEST_ASSERT_NOT_NULL(dm);
    dv = (float*)g_cuda.device_alloc(C*sizeof(float)); TEST_ASSERT_NOT_NULL(dv);
    dd = (float*)g_cuda.device_alloc(C*hw*sizeof(float)); TEST_ASSERT_NOT_NULL(dd);
    if (g_cuda.memcpy_h2d(dx, x, C*hw*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(dg, g, C*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(db, b, C*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(dm, m, C*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.memcpy_h2d(dv, v, C*sizeof(float), 0) != 0) TEST_FAIL();
    batchnorm_params_t bp={.C=C,.epsilon=1e-5f};
    const void* inputs[]={dx,dg,db,dm,dv,&hw}; void* outputs[]={dd};
    if (op->func(inputs,outputs,(const operator_params_t*)&bp,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(o, dd, C*hw*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_FLOAT_WITHIN(0.01f,1.0f,o[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,2.0f,o[1]);
    g_cuda.device_free(dx);g_cuda.device_free(dg);g_cuda.device_free(db);g_cuda.device_free(dm);g_cuda.device_free(dv);g_cuda.device_free(dd);
}

void test_sigmoid_f32_cuda(void) {
    const operator_registry_t* op = operator_find("sigmoid_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t n=4;
    float hi[4]={-1,0,1,2},ho[4]={0};
    float *di,*dd;
    di = (float*)g_cuda.device_alloc(n*sizeof(float)); TEST_ASSERT_NOT_NULL(di);
    dd = (float*)g_cuda.device_alloc(n*sizeof(float)); TEST_ASSERT_NOT_NULL(dd);
    if (g_cuda.memcpy_h2d(di, hi, n*sizeof(float), 0) != 0) TEST_FAIL();
    const void* inputs[]={di,&n}; void* outputs[]={dd};
    if (op->func(inputs,outputs,NULL,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(ho, dd, n*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_FLOAT_WITHIN(0.01f,0.2689f,ho[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,0.5f,ho[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,0.7311f,ho[2]);
    g_cuda.device_free(di);g_cuda.device_free(dd);
}

void test_gelu_f32_cuda(void) {
    const operator_registry_t* op = operator_find("gelu_f32_cuda");
    TEST_ASSERT_NOT_NULL(op);
    int64_t n=4;
    float hi[4]={-1,0,1,2},ho[4]={0};
    float *di,*dd;
    di = (float*)g_cuda.device_alloc(n*sizeof(float)); TEST_ASSERT_NOT_NULL(di);
    dd = (float*)g_cuda.device_alloc(n*sizeof(float)); TEST_ASSERT_NOT_NULL(dd);
    if (g_cuda.memcpy_h2d(di, hi, n*sizeof(float), 0) != 0) TEST_FAIL();
    const void* inputs[]={di,&n}; void* outputs[]={dd};
    if (op->func(inputs,outputs,NULL,NULL) != 0) TEST_FAIL();
    if (g_cuda.memcpy_d2h(ho, dd, n*sizeof(float), 0) != 0) TEST_FAIL();
    if (g_cuda.stream_synchronize(0) != 0) TEST_FAIL();
    TEST_ASSERT_FLOAT_WITHIN(0.01f,-0.158f,ho[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,0.0f,ho[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,0.841f,ho[2]);
    g_cuda.device_free(di);g_cuda.device_free(dd);
}

int main(void) {
    printf("A\n"); fflush(stdout);
    platform_init();
    cuda_platform_init(0);
    operator_init_all();
    printf("B\n"); fflush(stdout);
    UNITY_BEGIN();
    printf("C\n"); fflush(stdout);
    RUN_TEST(test_relu_f32_cuda);
    RUN_TEST(test_matmul_f32_cuda);
    RUN_TEST(test_conv2d_f32_cuda);
    RUN_TEST(test_maxpool2d_f32_cuda);
    RUN_TEST(test_avgpool2d_f32_cuda);
    RUN_TEST(test_batchnorm_f32_cuda);
    RUN_TEST(test_sigmoid_f32_cuda);
    RUN_TEST(test_gelu_f32_cuda);
    int result = UNITY_END();
    cuda_platform_finalize();
    platform_finalize();
    return result;
}
