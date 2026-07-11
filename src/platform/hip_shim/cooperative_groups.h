/* cooperative_groups.h — Shim for HIP compilation */
#ifndef COOPERATIVE_GROUPS_H_SHIM_
#define COOPERATIVE_GROUPS_H_SHIM_
#ifdef __HIPCC__
#include <hip/hip_cooperative_groups.h>
#else
#error "This shim should only be used with hipcc"
#endif
#endif
