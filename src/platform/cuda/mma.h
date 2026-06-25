/* mma.h — Shim for HIP compilation (WMMA Tensor Core API)
 * Note: rocWMMA is the HIP equivalent of nvcuda::wmma.
 * This shim is a placeholder; WMMA code paths are conditionally
 * compiled out under __HIPCC__ using tiled fallback kernels.
 */
#ifndef MMA_H_SHIM_
#define MMA_H_SHIM_
#ifdef __HIPCC__
/* WMMA not available via shim — code must use #ifdef __CUDACC__ guards */
#else
#error "This shim should only be used with hipcc"
#endif
#endif
