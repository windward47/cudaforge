#include "platform.h"

static const data_type_info_t s_type_info[DATA_TYPE_COUNT] = {
    [DATA_TYPE_F32]  = {"f32",  4, 1, 1},
    [DATA_TYPE_F16]  = {"f16",  2, 1, 1},
    [DATA_TYPE_BF16] = {"bf16", 2, 1, 1},
    [DATA_TYPE_I32]  = {"i32",  4, 0, 1},
    [DATA_TYPE_I8]   = {"i8",   1, 0, 1},
    [DATA_TYPE_U8]   = {"u8",   1, 0, 0},
    [DATA_TYPE_I64]  = {"i64",  8, 0, 1},
};

const data_type_info_t* data_type_get_info(data_type_t dt) {
    if (dt < 0 || dt >= DATA_TYPE_COUNT) return NULL;
    return &s_type_info[dt];
}
