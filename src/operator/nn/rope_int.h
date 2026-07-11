#ifndef ROPE_INT_H_
#define ROPE_INT_H_

#include <stdint.h>

/* RoPE 旋转布局
   INTERLEAVED: GPT-J 相邻对 (2i, 2i+1)        [cs cs cs cs]
   HALFSPLIT:   GPT-NeoX/LLaMA 前后半段 (i, i+d/2) [cc ss cc ss] (按半段)
   两者数学等价（都是 d/2 个二维子平面旋转），但权重按特定布局训练，
   推理必须匹配训练布局，否则结果错误。 */
#define ROPE_LAYOUT_INTERLEAVED 0
#define ROPE_LAYOUT_HALFSPLIT   1

typedef struct {
    int64_t seq_len;        /* S */
    int64_t head_dim;       /* d (must be even) */
    int64_t num_heads;      /* H */
    float   base;           /* theta base, typically 10000.0; used when inv_freq == NULL */
    int32_t layout;         /* ROPE_LAYOUT_INTERLEAVED or ROPE_LAYOUT_HALFSPLIT */
    const float* inv_freq;  /* 预计算频率表 [d/2], inv_freq[i] = 1/base^(2i/d);
                               NULL = 从 base 现算（向后兼容）。
                               频率表驱动架构：host 端按方案生成此表，
                               可支持 Linear/NTK/Dynamic NTK/YaRN 等上下文扩展变体。 */
    int64_t batch_size;     /* B, 默认 1; >1 时对 (B,S,H,d) 张量按 batch 独立旋转 */
    int64_t pos_offset;     /* 起始位置偏移, 默认 0; 适配 KV-cache 续写时
                               pos = pos_offset + 局部索引 (decode 单 token 时 = cache_len) */
} rope_params_t;

#endif /* ROPE_INT_H_ */
