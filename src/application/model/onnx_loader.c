#include "onnx_loader.h"
#include "onnx_parser.h"
#include "operator.h"
#include "matmul_int.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "batchnorm_int.h"
#include "add_int.h"
#include "reshape_int.h"
#include "globalavgpool_int.h"
#include "softmax_int.h"
#include "mul_int.h"
#include "concat_int.h"
#include "resize_int.h"
#include "transpose_int.h"
#include "sub_int.h"
#include "div_int.h"
#include "slice_int.h"
#include "split_int.h"
#include "layernorm_int.h"
#include "gather_int.h"
#include "squeeze_unsqueeze_int.h"
#include "reduce_int.h"
#include "cast_int.h"
#include "argmax_int.h"
#include "where_int.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char* dup_str(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* d = (char*)malloc(len);
    return d ? (char*)memcpy(d, s, len) : NULL;
}

/* ============================================================
 * ONNX protobuf field numbers (subset)
 * ============================================================ */
enum {
    F_ModelProto_graph          = 7,
    F_ModelProto_opset_import   = 8,  /* repeated OperatorSetIdProto */
    F_OperatorSetIdProto_version = 2, /* int64 */
    F_GraphProto_node           = 1,
    F_GraphProto_initializer    = 5,
    F_GraphProto_input          = 11,
    F_GraphProto_output         = 12,
    F_GraphProto_value_info     = 13,
    F_NodeProto_input           = 1,
    F_NodeProto_output          = 2,
    F_NodeProto_name            = 3,
    F_NodeProto_op_type         = 4,
    F_NodeProto_attribute       = 5,
    F_AttributeProto_name       = 1,
    F_AttributeProto_f          = 2,  /* float */
    F_AttributeProto_i          = 3,  /* int64 */
    F_AttributeProto_s          = 4,  /* string */
    F_AttributeProto_ints       = 8,  /* repeated int64 */
    F_TensorProto_dims          = 1,
    F_TensorProto_data_type     = 2,
    F_TensorProto_float_data    = 6,
    F_TensorProto_raw_data      = 9,
    F_TensorProto_external_data = 13,
    F_ValueInfoProto_name       = 1,
    F_ValueInfoProto_type       = 5,
    F_TypeProto_tensor_type     = 1,
    F_TensorShapeProto_dim      = 1,
    F_DimValue                  = 1,
};

#define ONNX_DTYPE_FLOAT  1
#define ONNX_DTYPE_INT64  7
#define MAX_TENSOR_NAME   128
#define MAX_NODES         512
#define MAX_TENSOR_INFOS  1024
#define TENSOR_HT_SIZE    2039  /* prime > 2 * MAX_TENSOR_INFOS for open addressing */

/* ============================================================
 * Internal: tensor info during parsing
 * ============================================================ */
typedef struct {
    char     name[MAX_TENSOR_NAME];
    int      ndim;
    int64_t  shape[8];
    int      dtype_onnx;
    uint8_t* raw_data;
    size_t   raw_data_size;
    int      is_initializer;
    int      tensor_id;  /* graph tensor ID after conversion */
} onnx_tensor_info_t;

/* ============================================================
 * Internal: parsed ONNX node before graph conversion
 * ============================================================ */
typedef struct {
    char     op_type[64];
    int      num_inputs;
    char     input_names[8][MAX_TENSOR_NAME];
    int      num_outputs;
    char     output_names[4][MAX_TENSOR_NAME];
    pb_field_t* attributes[16];
    int      num_attributes;
} onnx_node_info_t;

/* File-static: model file directory for resolving external data paths */
static char g_model_dir[512] = "";

/* File-static: opset version (default 11, parsed from ModelProto.opset_import) */
static int64_t g_model_opset = 11;

/* ============================================================
 * Internal: parsed ONNX model before graph construction
 * ============================================================ */
typedef struct {
    onnx_tensor_info_t tensors[MAX_TENSOR_INFOS];
    int                num_tensors;
    /* Open-addressing hash table: name → tensor index (TENSOR_HT_SIZE entries, -1 = empty) */
    int                tensor_ht[TENSOR_HT_SIZE];
    onnx_node_info_t   nodes[MAX_NODES];
    int                num_nodes;
    char               graph_input_names[16][MAX_TENSOR_NAME];
    int                num_graph_inputs;
    char               graph_output_names[16][MAX_TENSOR_NAME];
    int                num_graph_outputs;
} onnx_parsed_model_t;

/* ============================================================
 * FNV-1a hash for tensor name → hash table index
 * ============================================================ */
static unsigned int tensor_name_hash(const char* s) {
    unsigned int h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h % TENSOR_HT_SIZE;
}

/* ============================================================
 * Helper: find tensor by name (hash table, O(1) expected)
 * ============================================================ */
static onnx_tensor_info_t* find_tensor(onnx_parsed_model_t* m, const char* name) {
    unsigned int idx = tensor_name_hash(name);
    while (m->tensor_ht[idx] >= 0) {
        int ti = m->tensor_ht[idx];
        if (strcmp(m->tensors[ti].name, name) == 0)
            return &m->tensors[ti];
        idx = (idx + 1) % TENSOR_HT_SIZE;  /* linear probe */
    }
    return NULL;
}

static onnx_tensor_info_t* add_tensor(onnx_parsed_model_t* m, const char* name) {
    if (m->num_tensors >= MAX_TENSOR_INFOS) return NULL;
    onnx_tensor_info_t* t = &m->tensors[m->num_tensors];
    memset(t, 0, sizeof(*t));
    { size_t nl = strlen(name); if (nl >= MAX_TENSOR_NAME) nl = MAX_TENSOR_NAME - 1;
      memcpy(t->name, name, nl); t->name[nl] = '\0'; }
    t->tensor_id = -1;

    /* Insert into hash table */
    unsigned int idx = tensor_name_hash(t->name);
    while (m->tensor_ht[idx] >= 0)
        idx = (idx + 1) % TENSOR_HT_SIZE;  /* linear probe */
    m->tensor_ht[idx] = m->num_tensors;
    m->num_tensors++;
    return t;
}

/* ============================================================
 * Parse TensorProto → onnx_tensor_info_t
 * ============================================================ */
static int parse_tensor_proto(pb_message_t* msg, onnx_tensor_info_t* info) {
    /* Dims: try packed first, then non-packed */
    int64_t dims[8] = {0};
    int ndim = 0;

    pb_field_t* dims_field = pb_find_field(msg, F_TensorProto_dims);
    if (dims_field && dims_field->wire_type == PB_WIRE_LENGTH_DELIMITED) {
        pb_decode_packed_varints(dims_field->length_delimited.data,
                                  dims_field->length_delimited.size,
                                  dims, 8, &ndim);
    }
    /* Fallback: non-packed repeated varint fields */
    if (ndim == 0) {
        pb_field_t** all_dims = NULL;
        int dc = 0;
        pb_find_all_fields(msg, F_TensorProto_dims, &all_dims, &dc);
        for (int i = 0; i < dc && i < 8; i++) {
            dims[i] = (int64_t)all_dims[i]->varint_value;
        }
        ndim = (dc < 8) ? dc : 8;
        free(all_dims);
    }

    info->ndim = ndim;
    for (int i = 0; i < ndim; i++) info->shape[i] = dims[i];

    info->dtype_onnx = pb_field_get_int32(msg, F_TensorProto_data_type, 0);

    /* Raw data (field 9) takes priority */
    size_t raw_len = 0;
    const uint8_t* raw = pb_field_get_string(msg, F_TensorProto_raw_data, &raw_len);
    if (raw && raw_len > 0) {
        info->raw_data = (uint8_t*)malloc(raw_len);
        if (info->raw_data) {
            memcpy(info->raw_data, raw, raw_len);
            info->raw_data_size = raw_len;
        }
        return 0;
    }
    /* Fallback: try field 13 as raw_data (proto3 stores raw_data at field 13).
       Only use if it looks like valid tensor data (not external_data entries). */
    {
        size_t raw13_len = 0;
        const uint8_t* raw13 = pb_field_get_string(msg, 13, &raw13_len);
        if (raw13 && raw13_len > 0) {
            /* Heuristic: external_data entries are typically small (< 100 bytes)
               and contain key-value pairs. Raw tensor data is typically larger.
               Also check if the length matches expected tensor size. */
            int64_t expected_bytes = 1;
            for (int d = 0; d < info->ndim; d++) expected_bytes *= info->shape[d];
            expected_bytes *= 4; /* float32 */
            if (raw13_len >= (size_t)expected_bytes || raw13_len >= 256) {
                info->raw_data = (uint8_t*)malloc(raw13_len);
                if (info->raw_data) {
                    memcpy(info->raw_data, raw13, raw13_len);
                    info->raw_data_size = raw13_len;
                }
                return 0;
            }
        }
    }

    /* float_data: proto3 uses field 4 */
    pb_field_t* fd = pb_find_field(msg, 4);
    if (!fd || fd->wire_type != PB_WIRE_LENGTH_DELIMITED)
        fd = pb_find_field(msg, F_TensorProto_float_data);
    if (fd && fd->wire_type == PB_WIRE_LENGTH_DELIMITED) {
        size_t nf = fd->length_delimited.size / 4;
        info->raw_data = (uint8_t*)malloc(nf * sizeof(float));
        if (info->raw_data) {
            int fc = 0;
            pb_decode_float_data(fd->length_delimited.data,
                                  fd->length_delimited.size,
                                  (float*)info->raw_data, (int)nf, &fc);
            info->raw_data_size = (size_t)fc * sizeof(float);
        }
        return 0;
    }

    /* external_data: repeated StringStringEntryProto (field 13).
       Each entry has key (field 1) and value (field 2).
       Typically: key="location"→filename, key="offset"→byte_offset,
       key="length"→byte_length. */
    {
        pb_field_t** ed_fields = NULL;
        int ed_count = 0;
        pb_find_all_fields(msg, F_TensorProto_external_data, &ed_fields, &ed_count);
        if (ed_count > 0) {
            char location[256] = "";
            int64_t offset = 0;
            int64_t length = 0;

            for (int ei = 0; ei < ed_count; ei++) {
                pb_message_t* entry = pb_field_as_message(ed_fields[ei], 2);
                if (entry) {
                    size_t klen = 0, vlen = 0;
                    const uint8_t* key = pb_field_get_string(entry, 1, &klen);
                    const uint8_t* val = pb_field_get_string(entry, 2, &vlen);
                    if (key && val && klen > 0 && vlen > 0) {
                        if (klen == 8 && memcmp(key, "location", 8) == 0) {
                            int cp = (int)(vlen < sizeof(location)-1 ? vlen : sizeof(location)-1);
                            memcpy(location, val, cp);
                            location[cp] = '\0';
                        } else if (klen == 6 && memcmp(key, "offset", 6) == 0) {
                            char buf[32];
                            int n = (int)(vlen < sizeof(buf)-1 ? vlen : sizeof(buf)-1);
                            memcpy(buf, val, n); buf[n] = '\0';
                            offset = strtoll(buf, NULL, 10);
                        } else if (klen == 6 && memcmp(key, "length", 6) == 0) {
                            char buf[32];
                            int n = (int)(vlen < sizeof(buf)-1 ? vlen : sizeof(buf)-1);
                            memcpy(buf, val, n); buf[n] = '\0';
                            length = strtoll(buf, NULL, 10);
                        }
                    }
                    pb_message_destroy(entry);
                }
            }

            if (location[0] && length > 0) {
                char full_path[512];
                int n = snprintf(full_path, sizeof(full_path), "%s%s",
                                 g_model_dir, location);
                if (n > 0 && n < (int)sizeof(full_path)) {
                    FILE* ext = fopen(full_path, "rb");
                    if (ext) {
                        if (fseek(ext, (long)offset, SEEK_SET) == 0) {
                            info->raw_data = (uint8_t*)malloc((size_t)length);
                            if (info->raw_data) {
                                size_t r = fread(info->raw_data, 1, (size_t)length, ext);
                                if (r == (size_t)length) {
                                    info->raw_data_size = (size_t)length;
                                } else {
                                    fprintf(stderr, "onnx_load: tensor '%s' external data fread "
                                            "short read: got %zu, expected %lld\n",
                                            info->name, r, (long long)length);
                                    free(info->raw_data);
                                    info->raw_data = NULL;
                                }
                            } else {
                                fprintf(stderr, "onnx_load: tensor '%s' external data OOM "
                                        "(%lld bytes)\n", info->name, (long long)length);
                            }
                        } else {
                            fprintf(stderr, "onnx_load: tensor '%s' external data fseek failed "
                                    "(offset=%lld)\n", info->name, (long long)offset);
                        }
                        fclose(ext);
                    } else {
                        fprintf(stderr, "onnx_load: tensor '%s' external data file not found: "
                                "'%s'\n", info->name, full_path);
                    }
                } else {
                    fprintf(stderr, "onnx_load: tensor '%s' external data path too long: "
                            "'%s%s'\n", info->name, g_model_dir, location);
                }
            } else if (ed_count > 0) {
                fprintf(stderr, "onnx_load: tensor '%s' has external_data entries (%d) but "
                        "missing location or length\n", info->name, ed_count);
            }
        }
        free(ed_fields);
    }
    return 0;
}

/* ============================================================
 * Parse ONNX attribute → extract typed value
 * ============================================================ */
static int64_t attr_get_int(pb_message_t* attr, int64_t def) {
    return pb_field_get_int64(attr, F_AttributeProto_i, def);
}

static float attr_get_float(pb_message_t* attr, float def) {
    return pb_field_get_float(attr, F_AttributeProto_f, def);
}

static int attr_get_ints(pb_message_t* attr, int64_t* out, int max_n) {
    /* Try packed (length-delimited) first */
    pb_field_t* f = pb_find_field(attr, F_AttributeProto_ints);
    if (f && f->wire_type == PB_WIRE_LENGTH_DELIMITED) {
        int cnt = 0;
        pb_decode_packed_varints(f->length_delimited.data,
                                  f->length_delimited.size, out, max_n, &cnt);
        return cnt;
    }
    /* Fallback: non-packed repeated varint fields */
    pb_field_t** all = NULL;
    int ac = 0;
    pb_find_all_fields(attr, F_AttributeProto_ints, &all, &ac);
    int n = ac < max_n ? ac : max_n;
    for (int i = 0; i < n; i++) {
        out[i] = (int64_t)all[i]->varint_value;
    }
    free(all);
    return n;
}

/* Find attribute by name in a node's attribute list */
static pb_message_t* node_find_attr(onnx_node_info_t* node, const char* name) {
    for (int i = 0; i < node->num_attributes; i++) {
        size_t alen = 0;
        const uint8_t* aname = pb_field_get_string(
            pb_field_as_message(node->attributes[i], 2),
            F_AttributeProto_name, &alen);
        if (aname && alen == strlen(name) && memcmp(aname, name, alen) == 0) {
            return pb_field_as_message(node->attributes[i], 2);
        }
    }
    return NULL;
}

static int64_t node_attr_int(onnx_node_info_t* node, const char* name, int64_t def) {
    pb_message_t* a = node_find_attr(node, name);
    if (!a) return def;
    int64_t v = attr_get_int(a, def);
    pb_message_destroy(a);
    return v;
}

static float node_attr_float(onnx_node_info_t* node, const char* name, float def) {
    pb_message_t* a = node_find_attr(node, name);
    if (!a) return def;
    float v = attr_get_float(a, def);
    pb_message_destroy(a);
    return v;
}

static int node_attr_ints(onnx_node_info_t* node, const char* name,
                           int64_t* out, int max_n) {
    pb_message_t* a = node_find_attr(node, name);
    if (!a) return 0;
    int n = attr_get_ints(a, out, max_n);
    pb_message_destroy(a);
    return n;
}

static int node_attr_string_copy(onnx_node_info_t* node, const char* name,
                                  char* buf, int buf_size) {
    pb_message_t* a = node_find_attr(node, name);
    if (!a) return 0;
    size_t slen = 0;
    const uint8_t* s = pb_field_get_string(a, F_AttributeProto_s, &slen);
    if (!s || slen == 0) { pb_message_destroy(a); return 0; }
    int n = (int)(slen < (size_t)(buf_size - 1) ? slen : (size_t)(buf_size - 1));
    memcpy(buf, s, n);
    buf[n] = '\0';
    pb_message_destroy(a);
    return n;
}

/* ============================================================
 * Parse NodeProto → onnx_node_info_t
 * ============================================================ */
static int parse_node_proto(pb_message_t* msg, onnx_node_info_t* info) {
    memset(info, 0, sizeof(*info));

    /* op_type (field 4) */
    size_t oplen = 0;
    const uint8_t* op = pb_field_get_string(msg, F_NodeProto_op_type, &oplen);
    if (!op) return -1;
    int n_op = (int)(oplen < 63 ? oplen : 63);
    memcpy(info->op_type, op, n_op);
    info->op_type[n_op] = '\0';

    /* inputs (repeated field 1) */
    pb_field_t** in_fields = NULL;
    int in_cnt = 0;
    pb_find_all_fields(msg, F_NodeProto_input, &in_fields, &in_cnt);
    info->num_inputs = in_cnt < 8 ? in_cnt : 8;
    for (int i = 0; i < info->num_inputs; i++) {
        size_t slen = in_fields[i]->length_delimited.size;
        int n = (int)(slen < MAX_TENSOR_NAME - 1 ? slen : MAX_TENSOR_NAME - 1);
        memcpy(info->input_names[i], in_fields[i]->length_delimited.data, n);
        info->input_names[i][n] = '\0';
    }
    free(in_fields);

    /* outputs (repeated field 2) */
    pb_field_t** out_fields = NULL;
    int out_cnt = 0;
    pb_find_all_fields(msg, F_NodeProto_output, &out_fields, &out_cnt);
    info->num_outputs = out_cnt < 4 ? out_cnt : 4;
    for (int i = 0; i < info->num_outputs; i++) {
        size_t slen = out_fields[i]->length_delimited.size;
        int n = (int)(slen < MAX_TENSOR_NAME - 1 ? slen : MAX_TENSOR_NAME - 1);
        memcpy(info->output_names[i], out_fields[i]->length_delimited.data, n);
        info->output_names[i][n] = '\0';
    }
    free(out_fields);

    /* attributes (repeated field 5) */
    pb_field_t** attrs = NULL;
    int ac = 0;
    pb_find_all_fields(msg, F_NodeProto_attribute, &attrs, &ac);
    info->num_attributes = ac < 16 ? ac : 16;
    for (int i = 0; i < info->num_attributes; i++) {
        info->attributes[i] = attrs[i];
    }
    free(attrs);

    return 0;
}

/* ============================================================
 * Parse ValueInfoProto → extract name and shape
 * ============================================================ */
static int parse_value_info_shape(pb_message_t* msg, char* name_buf, int name_sz,
                                   int64_t* shape, int* ndim) {
    /* name (field 1) */
    size_t nlen = 0;
    const uint8_t* nm = pb_field_get_string(msg, F_ValueInfoProto_name, &nlen);
    if (!nm) return -1;
    int n = (int)(nlen < name_sz - 1 ? nlen : name_sz - 1);
    memcpy(name_buf, nm, n);
    name_buf[n] = '\0';

    /* type: proto3 uses field 2, proto2 uses field 5 — try both */
    pb_field_t* tf = pb_find_field(msg, 2);
    if (!tf) tf = pb_find_field(msg, F_ValueInfoProto_type);
    if (!tf) return 0; /* no shape, just name */

    pb_message_t* tmsg = pb_field_as_message(tf, 3);
    if (!tmsg) return 0;

    pb_field_t* tt = pb_find_field(tmsg, F_TypeProto_tensor_type);
    if (!tt) { pb_message_destroy(tmsg); return 0; }

    pb_message_t* tensor_msg = pb_field_as_message(tt, 2);
    pb_message_destroy(tmsg);
    if (!tensor_msg) return 0;

    /* TypeProto.Tensor.shape is field 2 → TensorShapeProto */
    pb_field_t* shape_field = pb_find_field(tensor_msg, 2);
    if (!shape_field) { pb_message_destroy(tensor_msg); return 0; }

    pb_message_t* shape_msg = pb_field_as_message(shape_field, 2);
    pb_message_destroy(tensor_msg);
    if (!shape_msg) return 0;

    /* dims (repeated field 1 of TensorShapeProto) */
    *ndim = 0;
    pb_field_t** dims = NULL;
    int dc = 0;
    pb_find_all_fields(shape_msg, F_TensorShapeProto_dim, &dims, &dc);
    for (int i = 0; i < dc && i < 8; i++) {
        pb_message_t* dm = pb_field_as_message(dims[i], 2);
        if (dm) {
            shape[i] = pb_field_get_int64(dm, F_DimValue, 0);
            pb_message_destroy(dm);
        } else {
            shape[i] = 0;
        }
    }
    *ndim = dc < 8 ? dc : 8;
    free(dims);
    pb_message_destroy(shape_msg);
    return 0;
}

/* ============================================================
 * Map ONNX op type → internal op_type_t
 * ============================================================ */
static op_type_t map_onnx_op(const char* onnx_op) {
    if (strcmp(onnx_op, "Relu")  == 0) return OP_RELU;
    if (strcmp(onnx_op, "Sigmoid") == 0) return OP_SIGMOID;
    if (strcmp(onnx_op, "Gelu")  == 0) return OP_GELU;
    if (strcmp(onnx_op, "Conv")  == 0) return OP_CONV2D;
    if (strcmp(onnx_op, "MatMul") == 0) return OP_MATMUL;
    if (strcmp(onnx_op, "Gemm")  == 0) return OP_MATMUL;
    if (strcmp(onnx_op, "MaxPool") == 0) return OP_MAXPOOL2D;
    if (strcmp(onnx_op, "AveragePool") == 0) return OP_AVGPOOL2D;
    if (strcmp(onnx_op, "BatchNormalization") == 0) return OP_BATCHNORM;
    if (strcmp(onnx_op, "Add")     == 0) return OP_ADD;
    if (strcmp(onnx_op, "Reshape") == 0) return OP_RESHAPE;
    if (strcmp(onnx_op, "GlobalAveragePool") == 0) return OP_GLOBALAVGPOOL;
    if (strcmp(onnx_op, "ReduceMean") == 0) return OP_GLOBALAVGPOOL;
    if (strcmp(onnx_op, "Softmax")  == 0) return OP_SOFTMAX;
    if (strcmp(onnx_op, "SiLU")  == 0) return OP_SILU;
    if (strcmp(onnx_op, "Mul")   == 0) return OP_MUL;
    if (strcmp(onnx_op, "Concat") == 0) return OP_CONCAT;
    if (strcmp(onnx_op, "Resize") == 0) return OP_RESIZE;
    if (strcmp(onnx_op, "Transpose") == 0) return OP_TRANSPOSE;
    if (strcmp(onnx_op, "Sub")   == 0) return OP_SUB;
    if (strcmp(onnx_op, "Div")   == 0) return OP_DIV;
    if (strcmp(onnx_op, "Slice") == 0) return OP_SLICE;
    if (strcmp(onnx_op, "Split") == 0) return OP_SPLIT;
    if (strcmp(onnx_op, "LayerNormalization") == 0) return OP_LAYERNORM;
    if (strcmp(onnx_op, "Gather") == 0) return OP_GATHER;
    if (strcmp(onnx_op, "Squeeze") == 0) return OP_SQUEEZE_UNSQUEEZE;
    if (strcmp(onnx_op, "Unsqueeze") == 0) return OP_SQUEEZE_UNSQUEEZE;
    if (strcmp(onnx_op, "Exp")   == 0) return OP_EXP;
    if (strcmp(onnx_op, "ReduceSum")  == 0) return OP_REDUCE;
    if (strcmp(onnx_op, "ReduceMax")  == 0) return OP_REDUCE;
    if (strcmp(onnx_op, "Cast")       == 0) return OP_CAST;
    if (strcmp(onnx_op, "ArgMax")     == 0) return OP_ARGMAX;
    if (strcmp(onnx_op, "Where")     == 0) return OP_WHERE;
    if (strcmp(onnx_op, "Tanh")     == 0) return OP_TANH;
    return (op_type_t)(-1);
}

/* ============================================================
 * Shape inference for a node
 * ============================================================ */
static int infer_output_shape(onnx_node_info_t* node, onnx_parsed_model_t* model) {
    op_type_t ot = map_onnx_op(node->op_type);
    if ((int)ot < 0) return -1;

    /* Get input 0 shape */
    onnx_tensor_info_t* in0 = find_tensor(model, node->input_names[0]);
    if (!in0) return 0; /* unknown input shape — skip */

    /* Elementwise ops: output = input shape */
    if (ot == OP_RELU || ot == OP_SIGMOID || ot == OP_GELU || ot == OP_SILU
        || ot == OP_BATCHNORM || ot == OP_ADD || ot == OP_MUL || ot == OP_SOFTMAX
        || ot == OP_SUB || ot == OP_DIV || ot == OP_EXP || ot == OP_CAST
        || ot == OP_TANH) {
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = in0->ndim;
                for (int d = 0; d < in0->ndim; d++) out->shape[d] = in0->shape[d];
            }
        }
        return 0;
    }

    /* Where: output = broadcast shape of condition, X, Y */
    if (ot == OP_WHERE) {
        int64_t out_shape[8] = {0};
        int out_ndim = 0;
        for (int ii = 0; ii < node->num_inputs && ii < 3; ii++) {
            onnx_tensor_info_t* inp = find_tensor(model, node->input_names[ii]);
            if (!inp) continue;
            if (inp->ndim > out_ndim) out_ndim = inp->ndim;
            for (int d = 0; d < inp->ndim; d++) {
                int od = out_ndim - inp->ndim + d;  /* right-align */
                if (od >= 0 && inp->shape[d] > out_shape[od])
                    out_shape[od] = inp->shape[d];
            }
        }
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = out_ndim;
                for (int d = 0; d < out_ndim; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Conv / Pooling */
    if (ot == OP_CONV2D || ot == OP_MAXPOOL2D || ot == OP_AVGPOOL2D) {
        onnx_tensor_info_t* w = find_tensor(model, node->input_names[1]);
        if (!w || w->ndim < 4) return 0;

        int64_t N  = in0->ndim >= 4 ? in0->shape[0] : 1;
        int64_t C  = in0->ndim >= 4 ? in0->shape[1] : 1;
        int64_t H  = in0->ndim >= 4 ? in0->shape[2] : 1;
        int64_t W  = in0->ndim >= 4 ? in0->shape[3] : 1;
        int64_t K  = ot == OP_CONV2D ? w->shape[0] : C;
        int64_t kH = w->shape[2];
        int64_t kW = w->shape[3];

        int64_t stride_h = node_attr_int(node, "strides", 1);
        int64_t stride_w = stride_h;
        int64_t ints[4];
        int ni = node_attr_ints(node, "strides", ints, 4);
        if (ni >= 2) { stride_h = ints[0]; stride_w = ints[1]; }

        int64_t dil_h = 1, dil_w = 1;
        ni = node_attr_ints(node, "dilations", ints, 4);
        if (ni >= 2) { dil_h = ints[0]; dil_w = ints[1]; }

        int64_t pad_h = 0, pad_w = 0;
        ni = node_attr_ints(node, "pads", ints, 4);
        if (ni >= 2) { pad_h = ints[0]; pad_w = ints[1]; }

        int64_t OH = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
        int64_t OW = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

        int64_t out_shape[4] = {N, K, OH, OW};
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = 4;
                for (int d = 0; d < 4; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* MatMul / Gemm */
    if (ot == OP_MATMUL) {
        onnx_tensor_info_t* in1 = find_tensor(model, node->input_names[1]);
        if (!in1) return 0;

        int64_t transB = node_attr_int(node, "transB", 0);
        int64_t M = in0->ndim >= 2 ? in0->shape[in0->ndim - 2] : 1;
        int64_t N = transB ? in1->shape[0] : (in1->ndim >= 2 ? in1->shape[in1->ndim - 1] : 1);

        /* Support batched MatMul: preserve batch dims */
        int ndim_out = (in0->ndim > in1->ndim) ? in0->ndim : in1->ndim;
        if (ndim_out < 2) ndim_out = 2;
        int64_t out_shape[8] = {0};
        out_shape[ndim_out - 2] = M;
        out_shape[ndim_out - 1] = N;
        for (int d = 0; d < ndim_out - 2; d++) {
            int64_t a_dim = (d < in0->ndim - 2) ? in0->shape[d] : 1;
            int64_t b_dim = (d < in1->ndim - 2) ? in1->shape[d] : 1;
            out_shape[d] = (a_dim > b_dim) ? a_dim : b_dim;
        }

        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_out;
                for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* GlobalAveragePool: NCHW → N C 1 1 */
    if (ot == OP_GLOBALAVGPOOL) {
        int64_t N = in0->ndim >= 4 ? in0->shape[0] : 1;
        int64_t C = in0->ndim >= 4 ? in0->shape[1] : 1;
        int64_t out_shape[4] = {N, C, 1, 1};
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = 4;
                for (int d = 0; d < 4; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* ReduceSum / ReduceMax: output shape = input shape with axes reduced */
    if (ot == OP_REDUCE) {
        int64_t axes[8];
        int num_axes = node_attr_ints(node, "axes", axes, 8);
        int64_t keepdims = node_attr_int(node, "keepdims", 1);

        /* If no axes attribute, read from input[1] initializer */
        if (num_axes == 0 && node->num_inputs >= 2 && node->input_names[1][0] != '\0') {
            onnx_tensor_info_t* ax_info = find_tensor(model, node->input_names[1]);
            if (ax_info && ax_info->raw_data) {
                int64_t* av = (int64_t*)ax_info->raw_data;
                num_axes = (int)(ax_info->raw_data_size / sizeof(int64_t));
                for (int i = 0; i < num_axes && i < 8; i++) axes[i] = av[i];
            }
        }

        /* Normalize negative axes */
        for (int i = 0; i < num_axes; i++) {
            if (axes[i] < 0) axes[i] += in0->ndim;
        }

        /* Build boolean mask of which axes are reduced */
        int reduced[8] = {0};
        for (int i = 0; i < num_axes && i < 8; i++) {
            if (axes[i] >= 0 && axes[i] < 8) reduced[axes[i]] = 1;
        }

        if (keepdims) {
            int64_t out_shape[8];
            int ndim_out = in0->ndim;
            for (int d = 0; d < in0->ndim; d++)
                out_shape[d] = reduced[d] ? 1 : in0->shape[d];
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        } else {
            int64_t out_shape[8];
            int ndim_out = 0;
            for (int d = 0; d < in0->ndim; d++) {
                if (!reduced[d]) out_shape[ndim_out++] = in0->shape[d];
            }
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        }
        return 0;
    }

    /* ArgMax: output = input shape with axis removed (or set to 1 if keepdims) */
    if (ot == OP_ARGMAX) {
        int64_t axis = node_attr_int(node, "axis", 0);
        int64_t keepdims = node_attr_int(node, "keepdims", 1);
        if (axis < 0) axis += in0->ndim;

        if (keepdims) {
            int64_t out_shape[8];
            int ndim_out = in0->ndim;
            for (int d = 0; d < in0->ndim; d++)
                out_shape[d] = (d == axis) ? 1 : in0->shape[d];
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        } else {
            int64_t out_shape[8];
            int ndim_out = 0;
            for (int d = 0; d < in0->ndim; d++) {
                if (d != axis) out_shape[ndim_out++] = in0->shape[d];
            }
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        }
        return 0;
    }

    /* Reshape: output shape from input[1] (initializer or constant tensor).
       opset < 14: shape is typically an initializer.
       opset ≥ 14: shape can be any tensor with known values. */
    if (ot == OP_RESHAPE) {
        onnx_tensor_info_t* shape_info = find_tensor(model, node->input_names[1]);
        if (!shape_info || !shape_info->raw_data)
            return 0;
        int64_t* shape_raw = (int64_t*)shape_info->raw_data;
        int64_t numel = 1;
        for (int d = 0; d < shape_info->ndim; d++) numel *= shape_info->shape[d];
        int ndim_out = (int)numel;
        if (ndim_out > 8) ndim_out = 8;

        /* Compute total input elements for resolving -1 */
        int64_t input_total = 1;
        for (int d = 0; d < in0->ndim; d++) input_total *= in0->shape[d];

        int64_t out_shape[8];
        int64_t known_product = 1;
        int minus_one_idx = -1;
        for (int d = 0; d < ndim_out; d++) {
            out_shape[d] = shape_raw[d];
            if (shape_raw[d] == -1) {
                minus_one_idx = d;
            } else if (shape_raw[d] == 0 && d < in0->ndim) {
                out_shape[d] = in0->shape[d];
            }
            if (out_shape[d] > 0) known_product *= out_shape[d];
        }
        if (minus_one_idx >= 0 && known_product > 0) {
            out_shape[minus_one_idx] = input_total / known_product;
        }

        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_out;
                for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Transpose: permute dimensions */
    if (ot == OP_TRANSPOSE) {
        int64_t perm[8];
        int ndim_perm = node_attr_ints(node, "perm", perm, 8);
        int ndim = in0->ndim;
        if (ndim_perm == 0) {
            for (int d = 0; d < ndim; d++) perm[d] = ndim - 1 - d;
            ndim_perm = ndim;
        }
        int64_t out_shape[8];
        for (int d = 0; d < ndim_perm && d < ndim; d++)
            out_shape[d] = in0->shape[perm[d]];
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_perm;
                for (int d = 0; d < ndim_perm; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Concat: concatenate along axis */
    if (ot == OP_CONCAT) {
        int64_t axis = node_attr_int(node, "axis", 1);
        int ndim_out = in0->ndim;
        if (axis < 0) axis += ndim_out;
        int64_t out_shape[8];
        for (int d = 0; d < ndim_out; d++) out_shape[d] = in0->shape[d];
        out_shape[axis] = 0;
        for (int ii = 0; ii < node->num_inputs && ii < 8; ii++) {
            if (node->input_names[ii][0] == '\0') continue;
            onnx_tensor_info_t* in_i = find_tensor(model, node->input_names[ii]);
            if (in_i && in_i->ndim > axis)
                out_shape[axis] += in_i->shape[axis];
        }
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_out;
                for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Resize: nearest-neighbor upsample */
    if (ot == OP_RESIZE) {
        onnx_tensor_info_t* scales_info = NULL;
        if (node->num_inputs >= 3 && node->input_names[2][0] != '\0')
            scales_info = find_tensor(model, node->input_names[2]);
        if (!scales_info && node->num_inputs >= 2 && node->input_names[1][0] != '\0')
            scales_info = find_tensor(model, node->input_names[1]);
        float scale_h = 1.0f, scale_w = 1.0f;
        if (scales_info && scales_info->raw_data
            && scales_info->raw_data_size >= 4 * (int)sizeof(float)) {
            float* s = (float*)scales_info->raw_data;
            scale_h = s[2]; scale_w = s[3];
        }
        int64_t N  = in0->ndim >= 4 ? in0->shape[0] : 1;
        int64_t C  = in0->ndim >= 4 ? in0->shape[1] : 1;
        int64_t H  = in0->ndim >= 4 ? in0->shape[2] : 1;
        int64_t W  = in0->ndim >= 4 ? in0->shape[3] : 1;
        int64_t OH = (int64_t)(H * scale_h);
        int64_t OW = (int64_t)(W * scale_w);
        int64_t out_shape[4] = {N, C, OH, OW};
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = 4;
                for (int d = 0; d < 4; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Slice: compute output shape from starts/ends/steps initializers */
    if (ot == OP_SLICE) {
        int ndim = in0->ndim;
        if (ndim < 1 || ndim > 8) return 0;

        /* Initialize: full range */
        int64_t starts[8] = {0};
        int64_t ends[8] = {0};
        int64_t steps[8] = {1,1,1,1,1,1,1,1};
        int     slice_axes[8];
        int     num_axes = 0;
        for (int d = 0; d < ndim; d++) ends[d] = in0->shape[d];

        /* Read starts from input[1] initializer (length matches axes count, not ndim) */
        onnx_tensor_info_t* st = find_tensor(model, node->input_names[1]);
        if (st && st->raw_data && st->raw_data_size >= sizeof(int64_t)) {
            int64_t* sv = (int64_t*)st->raw_data;
            int nsv = (int)(st->raw_data_size / sizeof(int64_t));
            for (int d = 0; d < nsv && d < 8; d++) starts[d] = sv[d];
        }

        /* Read ends from input[2] initializer */
        onnx_tensor_info_t* en = find_tensor(model, node->input_names[2]);
        if (en && en->raw_data && en->raw_data_size >= sizeof(int64_t)) {
            int64_t* ev = (int64_t*)en->raw_data;
            int nev = (int)(en->raw_data_size / sizeof(int64_t));
            for (int d = 0; d < nev && d < 8; d++) ends[d] = ev[d];
        }

        /* Read axes from input[3] initializer (optional) */
        num_axes = 0;
        if (node->num_inputs >= 4 && node->input_names[3][0] != '\0') {
            onnx_tensor_info_t* ax = find_tensor(model, node->input_names[3]);
            if (ax && ax->raw_data) {
                int64_t* av = (int64_t*)ax->raw_data;
                int n = (int)(ax->raw_data_size / sizeof(int64_t));
                for (int i = 0; i < n && i < 8; i++) {
                    slice_axes[i] = (int)av[i];
                }
                num_axes = n;
            }
        }

        /* Read steps from input[4] initializer (optional) */
        if (node->num_inputs >= 5 && node->input_names[4][0] != '\0') {
            onnx_tensor_info_t* sp = find_tensor(model, node->input_names[4]);
            if (sp && sp->raw_data && sp->raw_data_size >= sizeof(int64_t)) {
                int64_t* sv2 = (int64_t*)sp->raw_data;
                int nsv2 = (int)(sp->raw_data_size / sizeof(int64_t));
                for (int d = 0; d < nsv2 && d < 8; d++) steps[d] = sv2[d];
            }
        }

        /* If axes specified, apply starts/ends/steps only to those axes */
        if (num_axes > 0) {
            int64_t full_starts[8] = {0};
            int64_t full_ends[8];
            int64_t full_steps[8] = {1,1,1,1,1,1,1,1};
            for (int d = 0; d < ndim; d++) full_ends[d] = in0->shape[d];
            for (int i = 0; i < num_axes && i < 8; i++) {
                int ax = slice_axes[i];
                if (ax >= 0 && ax < ndim) {
                    full_starts[ax] = starts[i];
                    full_ends[ax] = ends[i];
                    full_steps[ax] = steps[i];
                }
            }
            for (int d = 0; d < ndim; d++) {
                starts[d] = full_starts[d];
                ends[d] = full_ends[d];
                steps[d] = full_steps[d];
            }
        }

        /* Clamp to valid range and compute output shape */
        int64_t out_shape[8];
        int ndim_out = ndim;
        for (int d = 0; d < ndim; d++) {
            if (starts[d] < 0) starts[d] += in0->shape[d];
            if (ends[d] < 0) ends[d] += in0->shape[d];
            if (starts[d] < 0) starts[d] = 0;
            if (ends[d] > in0->shape[d]) ends[d] = in0->shape[d];
            out_shape[d] = (ends[d] - starts[d] + steps[d] - 1) / steps[d];
            if (out_shape[d] < 0) out_shape[d] = 0;
        }

        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_out;
                for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Split: compute per-output shapes */
    if (ot == OP_SPLIT) {
        int64_t axis = node_attr_int(node, "axis", 0);
        int ndim = in0->ndim;
        if (axis < 0) axis = ndim + axis;

        /* Read split sizes from input[1] initializer (optional) */
        int64_t splits[8];
        int num_splits = 0;
        if (node->num_inputs >= 2 && node->input_names[1][0] != '\0') {
            onnx_tensor_info_t* sp = find_tensor(model, node->input_names[1]);
            if (sp && sp->raw_data) {
                int64_t* sv = (int64_t*)sp->raw_data;
                int n = (int)(sp->raw_data_size / sizeof(int64_t));
                for (int i = 0; i < n && i < 8; i++) {
                    splits[i] = sv[i];
                }
                num_splits = n;
            }
        }

        /* If no explicit split sizes, divide equally */
        if (num_splits == 0) {
            num_splits = node->num_outputs;
            int64_t eq = in0->shape[axis] / num_splits;
            for (int i = 0; i < num_splits; i++) splits[i] = eq;
        }

        for (int oi = 0; oi < node->num_outputs && oi < num_splits; oi++) {
            int64_t out_shape[8];
            for (int d = 0; d < ndim; d++) out_shape[d] = in0->shape[d];
            out_shape[axis] = splits[oi];

            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim;
                for (int d = 0; d < ndim; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* LayerNormalization: output shape = input shape (elementwise-like) */
    if (ot == OP_LAYERNORM) {
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out && in0) {
                out->ndim = in0->ndim;
                for (int d = 0; d < in0->ndim; d++) out->shape[d] = in0->shape[d];
            }
        }
        return 0;
    }

    /* Gather: output shape = input shape with axis dim replaced by indices shape */
    if (ot == OP_GATHER) {
        onnx_tensor_info_t* indices = NULL;
        if (node->num_inputs >= 2 && node->input_names[1][0] != '\0')
            indices = find_tensor(model, node->input_names[1]);
        int64_t axis = node_attr_int(node, "axis", 0);
        int ndim_in = in0 ? in0->ndim : 1;
        if (axis < 0) axis += ndim_in;

        int ni = indices ? indices->ndim : 1;
        int ndim_out = ndim_in - 1 + ni;
        int64_t out_shape[8];
        int di = 0;
        for (int d = 0; d < axis && di < 8; d++)
            out_shape[di++] = in0->shape[d];
        if (indices) {
            for (int d = 0; d < ni && di < 8; d++)
                out_shape[di++] = indices->shape[d];
        } else {
            out_shape[di++] = 1;
        }
        for (int d = (int)axis + 1; d < ndim_in && di < 8; d++)
            out_shape[di++] = in0->shape[d];

        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = ndim_out;
                for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
            }
        }
        return 0;
    }

    /* Squeeze: remove axes of size 1; Unsqueeze: insert axes of size 1 */
    if (ot == OP_SQUEEZE_UNSQUEEZE) {
        int64_t axes[8];
        int num_axes = 0;
        int is_unsqueeze = (strcmp(node->op_type, "Unsqueeze") == 0);

        /* Read axes from attribute or input[1] (opset 13+) */
        num_axes = node_attr_ints(node, "axes", axes, 8);
        if (num_axes == 0 && node->num_inputs >= 2) {
            onnx_tensor_info_t* ax = find_tensor(model, node->input_names[1]);
            if (ax && ax->raw_data) {
                int64_t* av = (int64_t*)ax->raw_data;
                int n = (int)(ax->raw_data_size / sizeof(int64_t));
                for (int i = 0; i < n && i < 8; i++) axes[i] = av[i];
                num_axes = n;
            }
        }

        int ndim_in = in0 ? in0->ndim : 1;
        if (is_unsqueeze) {
            int ndim_out = ndim_in + num_axes;
            int64_t out_shape[8] = {0};
            /* Mark inserted axes */
            bool inserted[8] = {false};
            for (int i = 0; i < num_axes && i < 8; i++) {
                int64_t ax = axes[i];
                if (ax < 0) ax += ndim_out;
                if (ax >= 0 && ax < ndim_out) { out_shape[ax] = 1; inserted[ax] = true; }
            }
            /* Fill remaining with input dims */
            for (int d = 0, si = 0; d < ndim_out; d++) {
                if (!inserted[d] && si < ndim_in) out_shape[d] = in0->shape[si++];
            }
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        } else {
            /* Squeeze: remove axes of size 1 */
            int squeeze_axes[8];
            int nsq = num_axes;
            for (int i = 0; i < nsq; i++) {
                squeeze_axes[i] = (int)axes[i];
                if (squeeze_axes[i] < 0) squeeze_axes[i] += ndim_in;
            }
            if (nsq == 0) {
                /* Auto: remove all axes with size 1 */
                for (int d = 0; d < ndim_in && nsq < 8; d++)
                    if (in0->shape[d] == 1) squeeze_axes[nsq++] = d;
            }
            int64_t out_shape[8];
            int ndim_out = 0;
            for (int d = 0; d < ndim_in; d++) {
                bool squeeze = false;
                for (int s = 0; s < nsq; s++)
                    if (squeeze_axes[s] == d) { squeeze = true; break; }
                if (!squeeze) out_shape[ndim_out++] = in0->shape[d];
            }
            for (int oi = 0; oi < node->num_outputs; oi++) {
                onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
                if (!out) out = add_tensor(model, node->output_names[oi]);
                if (out) {
                    out->ndim = ndim_out;
                    for (int d = 0; d < ndim_out; d++) out->shape[d] = out_shape[d];
                }
            }
        }
        return 0;
    }

    return 0;
}

/* ============================================================
 * Build inference_graph_t from parsed ONNX model
 * ============================================================ */
static inference_graph_t* build_graph_from_onnx(onnx_parsed_model_t* pm) {
    inference_graph_t* g = graph_create();
    if (!g) return NULL;


    /* Create OP_INPUT nodes for graph inputs */
    int input_node_ids[16];
    int input_tensor_ids[16];
    for (int i = 0; i < pm->num_graph_inputs; i++) {
        onnx_tensor_info_t* t = find_tensor(pm, pm->graph_input_names[i]);
        if (!t) t = add_tensor(pm, pm->graph_input_names[i]);

        tensor_t* tt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
        int tid = graph_add_tensor(g, tt);
        t->tensor_id = tid;

        int num_in = 0, num_out = 1;
        int out_tids[] = {tid};
        int nid = graph_add_node(g, OP_INPUT, num_in, NULL, num_out, out_tids,
                                  0, NULL, NULL, 0);
        graph_set_input(g, nid);
        input_node_ids[i] = nid;
        input_tensor_ids[i] = tid;
    }

    /* Create tensors for all intermediate edges (non-input, non-initializer) */
    for (int ni = 0; ni < pm->num_nodes; ni++) {
        onnx_node_info_t* node = &pm->nodes[ni];
        op_type_t ot = map_onnx_op(node->op_type);
        if ((int)ot < 0) {
            fprintf(stderr, "WARNING: unsupported ONNX op '%s', skipping\n",
                    node->op_type);
            continue;
        }

        /* Ensure output tensors exist in graph */
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* t = find_tensor(pm, node->output_names[oi]);
            if (!t) t = add_tensor(pm, node->output_names[oi]);
            if (!t) continue;
            if (t->tensor_id < 0) {
                tensor_t* tt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
                if (!tt) continue;
                t->tensor_id = graph_add_tensor(g, tt);
            }
        }
    }


    /* Create OP_OUTPUT nodes for graph outputs */
    for (int i = 0; i < pm->num_graph_outputs; i++) {
        onnx_tensor_info_t* t = find_tensor(pm, pm->graph_output_names[i]);
        if (!t || t->tensor_id < 0) continue;

        int in_tids[] = {t->tensor_id};
        int nid = graph_add_node(g, OP_OUTPUT, 1, in_tids, 0, NULL,
                                  0, NULL, NULL, 0);
        graph_set_output(g, nid);
    }


    /* Create operator nodes */
    for (int ni = 0; ni < pm->num_nodes; ni++) {
        onnx_node_info_t* node = &pm->nodes[ni];
        op_type_t ot = map_onnx_op(node->op_type);
        if ((int)ot < 0) continue;

        /* Resolve input tensor IDs: data inputs only (skip initializers).
           Initializers are passed as weights, separate from the data input array. */
        int in_tids[8];
        int num_data_in = 0;
        tensor_t* weights[8];
        int num_weights = 0;

        for (int ii = 0; ii < node->num_inputs; ii++) {
            onnx_tensor_info_t* t = find_tensor(pm, node->input_names[ii]);

            /* Slice, Split, and Squeeze/Unsqueeze have int64 initializer inputs
               (starts/ends/axes/steps). Skip them here — already extracted to params. */
            if ((ot == OP_SLICE || ot == OP_SPLIT || ot == OP_SQUEEZE_UNSQUEEZE
                 || ot == OP_REDUCE)
                && t && t->is_initializer)
                continue;

            /* For non-commutative ops (Sub, Div, Gather), an initializer at
               position 0 must keep its position in op_inputs rather than being
               pushed to the end. Gather(data, indices) needs data at inputs[0]. */
            /* Determine if this input should be treated as a weight (embedded in node)
               or as a data input (resolved at execution time).
               Add/Mul/Sub/Div/Where: all inputs are data (broadcast semantics).
               Gather: input[0] is data (embedding table).
               Other ops: initializers become weights. */
            int as_weight = (t && t->is_initializer && t->raw_data
                             && ot != OP_ADD && ot != OP_MUL
                             && ot != OP_SUB && ot != OP_DIV
                             && ot != OP_WHERE
                             && !(ot == OP_GATHER && ii == 0));

            if (as_weight) {
                tensor_t* wt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
                if (wt && wt->data) {
                    /* Convert int64 initializers (e.g. Gather indices) to float */
                    if (t->dtype_onnx == ONNX_DTYPE_INT64 && t->raw_data) {
                        int64_t num = (int64_t)(t->raw_data_size / sizeof(int64_t));
                        if (num > wt->numel) num = wt->numel;
                        int64_t* src = (int64_t*)t->raw_data;
                        float* dst = (float*)wt->data;
                        for (int64_t j = 0; j < num; j++) dst[j] = (float)src[j];
                    } else if (t->dtype_onnx == 9 && t->raw_data) {
                        /* BOOL tensor: 1 byte per element, convert to float */
                        int64_t num = (int64_t)t->raw_data_size;
                        if (num > wt->numel) num = wt->numel;
                        uint8_t* src = (uint8_t*)t->raw_data;
                        float* dst = (float*)wt->data;
                        for (int64_t j = 0; j < num; j++) dst[j] = (src[j] != 0) ? 1.0f : 0.0f;
                    } else if (t->raw_data_size <= (size_t)wt->numel * sizeof(float)) {
                        memcpy(wt->data, t->raw_data, t->raw_data_size);
                    }
                }
                weights[num_weights++] = wt;
            } else {
                /* Ensure tensor exists in graph for data inputs */
                if (t && t->tensor_id < 0) {
                    tensor_t* tt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
                    if (tt) {
                        /* Copy initializer data into graph tensor */
                        if (t->is_initializer && t->raw_data) {
                            if (t->dtype_onnx == 9) {
                                /* BOOL: 1 byte per element */
                                int64_t num = (int64_t)t->raw_data_size;
                                if (num > tt->numel) num = tt->numel;
                                uint8_t* src = (uint8_t*)t->raw_data;
                                float* dst = (float*)tt->data;
                                for (int64_t j = 0; j < num; j++) dst[j] = (src[j] != 0) ? 1.0f : 0.0f;
                            } else if (t->raw_data_size <= (size_t)tt->numel * sizeof(float)) {
                                memcpy(tt->data, t->raw_data, t->raw_data_size);
                            }
                        }
                        t->tensor_id = graph_add_tensor(g, tt);
                    }
                }
                in_tids[num_data_in] = (t && t->tensor_id >= 0) ? t->tensor_id : -1;
                num_data_in++;
            }
        }

        /* Resolve output tensor IDs */
        int out_tids[4];
        int num_out = node->num_outputs;
        for (int oi = 0; oi < num_out; oi++) {
            onnx_tensor_info_t* t = find_tensor(pm, node->output_names[oi]);
            out_tids[oi] = (t && t->tensor_id >= 0) ? t->tensor_id : -1;
        }

        /* Build operator params */
        void* params = NULL;
        size_t params_size = 0;

        if (ot == OP_CONV2D) {
            conv_params_t cp;
            memset(&cp, 0, sizeof(cp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* w_t  = find_tensor(pm, node->input_names[1]);
            if (in_t && in_t->ndim >= 4) {
                cp.N = in_t->shape[0]; cp.C = in_t->shape[1];
                cp.H = in_t->shape[2]; cp.W = in_t->shape[3];
            }
            if (w_t && w_t->ndim >= 4) {
                cp.K = w_t->shape[0]; cp.kernel_h = w_t->shape[2]; cp.kernel_w = w_t->shape[3];
            }
            int64_t ints[4]; int na;
            cp.stride_h = 1; cp.stride_w = 1;
            na = node_attr_ints(node, "strides", ints, 4);
            if (na >= 2) { cp.stride_h = ints[0]; cp.stride_w = ints[1]; }
            cp.dilation_h = 1; cp.dilation_w = 1;
            na = node_attr_ints(node, "dilations", ints, 4);
            if (na >= 2) { cp.dilation_h = ints[0]; cp.dilation_w = ints[1]; }
            cp.pad_h = 0; cp.pad_w = 0;
            na = node_attr_ints(node, "pads", ints, 4);
            if (na >= 2) { cp.pad_h = ints[0]; cp.pad_w = ints[1]; }
            cp.groups = (int64_t)node_attr_int(node, "group", 1);
            params = &cp; params_size = sizeof(cp);
        } else if (ot == OP_MAXPOOL2D || ot == OP_AVGPOOL2D) {
            pool_params_t pp;
            memset(&pp, 0, sizeof(pp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            if (in_t && in_t->ndim >= 4) {
                pp.N = in_t->shape[0]; pp.C = in_t->shape[1];
                pp.H = in_t->shape[2]; pp.W = in_t->shape[3];
            }
            pp.kernel_h = (int64_t)node_attr_int(node, "kernel_shape", 1);
            pp.kernel_w = pp.kernel_h;
            int64_t ints2[4]; int na2;
            na2 = node_attr_ints(node, "kernel_shape", ints2, 4);
            if (na2 >= 2) { pp.kernel_h = ints2[0]; pp.kernel_w = ints2[1]; }
            pp.stride_h = 1; pp.stride_w = 1;
            na2 = node_attr_ints(node, "strides", ints2, 4);
            if (na2 >= 2) { pp.stride_h = ints2[0]; pp.stride_w = ints2[1]; }
            pp.pad_h = 0; pp.pad_w = 0;
            na2 = node_attr_ints(node, "pads", ints2, 4);
            if (na2 >= 2) { pp.pad_h = ints2[0]; pp.pad_w = ints2[1]; }
            params = &pp; params_size = sizeof(pp);
        } else if (ot == OP_MATMUL) {
            matmul_params_t mp;
            memset(&mp, 0, sizeof(mp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* w_t  = find_tensor(pm, node->input_names[1]);
            if (in_t && in_t->ndim >= 2) {
                mp.M = in_t->shape[0]; mp.K = in_t->shape[1];
            }
            int64_t transB = node_attr_int(node, "transB", 0);
            if (w_t && w_t->ndim >= 2) {
                mp.N = transB ? w_t->shape[0] : w_t->shape[1];
            }
            mp.transpose_a = node_attr_int(node, "transA", 0) != 0;
            mp.transpose_b = transB != 0;
            params = &mp; params_size = sizeof(mp);
        } else if (ot == OP_BATCHNORM) {
            batchnorm_params_t bp;
            memset(&bp, 0, sizeof(bp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            if (in_t && in_t->ndim >= 2) {
                bp.C = in_t->shape[1];
            }
            bp.epsilon = node_attr_float(node, "epsilon", 1e-5f);
            params = &bp; params_size = sizeof(bp);
        } else if (ot == OP_ADD) {
            add_params_t ap;
            memset(&ap, 0, sizeof(ap));
            onnx_tensor_info_t* in_a = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* in_b = find_tensor(pm, node->input_names[1]);
            if (in_a) { ap.numel = 1; for (int d = 0; d < in_a->ndim; d++) ap.numel *= in_a->shape[d]; }
            if (in_b) { ap.B_numel = 1; for (int d = 0; d < in_b->ndim; d++) ap.B_numel *= in_b->shape[d]; }
            params = &ap; params_size = sizeof(ap);
        } else if (ot == OP_RESHAPE) {
            reshape_params_t rp;
            memset(&rp, 0, sizeof(rp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* out_t = find_tensor(pm, node->output_names[0]);
            if (in_t) { rp.numel = 1; for (int d = 0; d < in_t->ndim; d++) rp.numel *= in_t->shape[d]; }
            if (out_t) {
                rp.ndim = out_t->ndim;
                for (int d = 0; d < out_t->ndim && d < MAX_TENSOR_DIMS; d++)
                    rp.shape[d] = out_t->shape[d];
            }
            params = &rp; params_size = sizeof(rp);
        } else if (ot == OP_GLOBALAVGPOOL) {
            globalavgpool_params_t gp;
            memset(&gp, 0, sizeof(gp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            if (in_t && in_t->ndim >= 4) {
                gp.N = in_t->shape[0]; gp.C = in_t->shape[1];
                gp.H = in_t->shape[2]; gp.W = in_t->shape[3];
            }
            params = &gp; params_size = sizeof(gp);
        } else if (ot == OP_SOFTMAX) {
            softmax_params_t sp;
            memset(&sp, 0, sizeof(sp));
            /* Softmax default axis: opset < 13 → 1, opset ≥ 13 → -1 */
            int64_t softmax_default_axis = (g_model_opset >= 13) ? -1 : 1;
            int64_t axis = node_attr_int(node, "axis", softmax_default_axis);
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            if (in_t) {
                /* Determine num_classes (axis dim) and num_blocks (product of other dims) */
                int64_t eff_axis = axis;
                if (eff_axis < 0) eff_axis += in_t->ndim;
                int64_t num_classes = (eff_axis >= 0 && eff_axis < in_t->ndim) ? in_t->shape[eff_axis] : 1;
                int64_t num_blocks = 1;
                for (int d = 0; d < in_t->ndim; d++) {
                    if (d != eff_axis) num_blocks *= in_t->shape[d];
                }
                sp.num_classes = num_classes;
                sp.num_blocks  = num_blocks;
            }
            params = &sp; params_size = sizeof(sp);
        } else if (ot == OP_MUL) {
            mul_params_t mp;
            memset(&mp, 0, sizeof(mp));
            onnx_tensor_info_t* in_a = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* in_b = find_tensor(pm, node->input_names[1]);
            if (in_a) { mp.numel = 1; for (int d = 0; d < in_a->ndim; d++) mp.numel *= in_a->shape[d]; }
            if (in_b) { mp.B_numel = 1; for (int d = 0; d < in_b->ndim; d++) mp.B_numel *= in_b->shape[d]; }
            params = &mp; params_size = sizeof(mp);
        } else if (ot == OP_WHERE) {
            where_params_t wp;
            memset(&wp, 0, sizeof(wp));
            /* Output numel from output tensor (set by shape inference) */
            onnx_tensor_info_t* out_w = find_tensor(pm, node->output_names[0]);
            if (out_w) { wp.numel = 1; for (int d = 0; d < out_w->ndim; d++) wp.numel *= out_w->shape[d]; }
            /* Per-input numel for broadcast */
            onnx_tensor_info_t* in_cond = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* in_x = find_tensor(pm, node->input_names[1]);
            onnx_tensor_info_t* in_y = find_tensor(pm, node->input_names[2]);
            if (in_cond) { wp.cond_numel = 1; for (int d = 0; d < in_cond->ndim; d++) wp.cond_numel *= in_cond->shape[d]; }
            if (in_x) { wp.x_numel = 1; for (int d = 0; d < in_x->ndim; d++) wp.x_numel *= in_x->shape[d]; }
            if (in_y) { wp.y_numel = 1; for (int d = 0; d < in_y->ndim; d++) wp.y_numel *= in_y->shape[d]; }
            params = &wp; params_size = sizeof(wp);
        } else if (ot == OP_CONCAT) {
            concat_params_t cp;
            memset(&cp, 0, sizeof(cp));
            cp.num_inputs = node->num_inputs;
            onnx_tensor_info_t* in0c = find_tensor(pm, node->input_names[0]);
            int ndim = in0c ? in0c->ndim : 0;
            cp.ndim = ndim;
            cp.axis = (int)node_attr_int(node, "axis", 1);
            if (cp.axis < 0) cp.axis += ndim;
            /* Compute outer (product of dims before axis) and inner (product after) */
            cp.outer = 1;
            for (int d = 0; d < cp.axis && d < ndim; d++) cp.outer *= in0c->shape[d];
            cp.inner = 1;
            for (int d = cp.axis + 1; d < ndim; d++) cp.inner *= in0c->shape[d];
            cp.C_total = 0;
            for (int ii = 0; ii < node->num_inputs && ii < 8; ii++) {
                onnx_tensor_info_t* in_i = find_tensor(pm, node->input_names[ii]);
                int64_t Ci = (in_i && in_i->ndim > cp.axis) ? in_i->shape[cp.axis] : 1;
                cp.C_offset[ii] = cp.C_total;
                cp.C_per_input[ii] = Ci;
                cp.C_total += Ci;
            }
            onnx_tensor_info_t* out_tc = find_tensor(pm, node->output_names[0]);
            if (out_tc) {
                cp.total_numel = 1;
                for (int d = 0; d < out_tc->ndim; d++) cp.total_numel *= out_tc->shape[d];
            }
            params = &cp; params_size = sizeof(cp);
        } else if (ot == OP_RESIZE) {
            resize_params_t rp;
            memset(&rp, 0, sizeof(rp));
            onnx_tensor_info_t* in_tr = find_tensor(pm, node->input_names[0]);
            if (in_tr && in_tr->ndim >= 4) {
                rp.N = in_tr->shape[0]; rp.C = in_tr->shape[1];
                rp.H_in = in_tr->shape[2]; rp.W_in = in_tr->shape[3];
            }
            onnx_tensor_info_t* scales_info = NULL;
            if (node->num_inputs >= 3 && node->input_names[2][0] != '\0')
                scales_info = find_tensor(pm, node->input_names[2]);
            if (!scales_info && node->num_inputs >= 2 && node->input_names[1][0] != '\0')
                scales_info = find_tensor(pm, node->input_names[1]);
            rp.mode = 0;  /* default: nearest */
            {
                char mode_str[32] = "";
                if (node_attr_string_copy(node, "mode", mode_str, sizeof(mode_str)) > 0) {
                    if (strcmp(mode_str, "linear") == 0) rp.mode = 1;
                }
            }
            rp.scale_h = 1.0f; rp.scale_w = 1.0f;
            if (scales_info && scales_info->raw_data
                && scales_info->raw_data_size >= 4 * (int)sizeof(float)) {
                float* s = (float*)scales_info->raw_data;
                rp.scale_h = s[2]; rp.scale_w = s[3];
            }
            onnx_tensor_info_t* out_tr = find_tensor(pm, node->output_names[0]);
            if (out_tr && out_tr->ndim >= 4) {
                rp.H_out = out_tr->shape[2]; rp.W_out = out_tr->shape[3];
            }
            params = &rp; params_size = sizeof(rp);
        } else if (ot == OP_TRANSPOSE) {
            transpose_params_t tp;
            memset(&tp, 0, sizeof(tp));
            onnx_tensor_info_t* in_tt = find_tensor(pm, node->input_names[0]);
            if (in_tt) {
                tp.ndim = in_tt->ndim;
                for (int d = 0; d < in_tt->ndim && d < 8; d++)
                    tp.shape[d] = in_tt->shape[d];
            }
            int nperm = node_attr_ints(node, "perm", tp.perm, 8);
            if (nperm == 0) {
                for (int d = 0; d < tp.ndim; d++)
                    tp.perm[d] = tp.ndim - 1 - d;
            }
            onnx_tensor_info_t* out_tt = find_tensor(pm, node->output_names[0]);
            if (out_tt) {
                for (int d = 0; d < out_tt->ndim && d < 8; d++)
                    tp.out_shape[d] = out_tt->shape[d];
            }
            params = &tp; params_size = sizeof(tp);
        } else if (ot == OP_SUB) {
            sub_params_t sp;
            memset(&sp, 0, sizeof(sp));
            onnx_tensor_info_t* in_a = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* in_b = find_tensor(pm, node->input_names[1]);
            if (in_a) { sp.numel = 1; for (int d = 0; d < in_a->ndim; d++) sp.numel *= in_a->shape[d]; }
            if (in_b) { sp.B_numel = 1; for (int d = 0; d < in_b->ndim; d++) sp.B_numel *= in_b->shape[d]; }
            params = &sp; params_size = sizeof(sp);
        } else if (ot == OP_DIV) {
            div_params_t dp;
            memset(&dp, 0, sizeof(dp));
            onnx_tensor_info_t* in_a = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* in_b = find_tensor(pm, node->input_names[1]);
            if (in_a) { dp.numel = 1; for (int d = 0; d < in_a->ndim; d++) dp.numel *= in_a->shape[d]; }
            if (in_b) { dp.B_numel = 1; for (int d = 0; d < in_b->ndim; d++) dp.B_numel *= in_b->shape[d]; }
            params = &dp; params_size = sizeof(dp);
        } else if (ot == OP_SLICE) {
            slice_params_t slp;
            memset(&slp, 0, sizeof(slp));
            onnx_tensor_info_t* in_sl = find_tensor(pm, node->input_names[0]);
            int ndim = in_sl ? in_sl->ndim : 0;
            slp.ndim = ndim;
            slp.in_numel = 1;
            int64_t ends[8];
            for (int d = 0; d < ndim; d++) {
                slp.in_shape[d] = in_sl->shape[d];
                slp.in_numel *= in_sl->shape[d];
                ends[d] = in_sl->shape[d];
                slp.steps[d] = 1;
                slp.starts[d] = 0;
            }
            /* Compute input strides */
            int64_t stride = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                slp.in_strides[d] = stride;
                stride *= slp.in_shape[d];
            }
            /* Read starts from input[1] initializer */
            onnx_tensor_info_t* sti = find_tensor(pm, node->input_names[1]);
            if (sti && sti->raw_data) {
                int64_t* sv = (int64_t*)sti->raw_data;
                for (int d = 0; d < ndim && d < (int)(sti->raw_data_size / sizeof(int64_t)); d++)
                    slp.starts[d] = sv[d];
            }
            /* Read ends from input[2] initializer */
            onnx_tensor_info_t* eni = find_tensor(pm, node->input_names[2]);
            if (eni && eni->raw_data) {
                int64_t* ev = (int64_t*)eni->raw_data;
                for (int d = 0; d < ndim && d < (int)(eni->raw_data_size / sizeof(int64_t)); d++)
                    ends[d] = ev[d];
            }
            /* Read axes from input[3] initializer (optional) */
            int slice_axes[8] = {0};
            int num_axes = 0;
            if (node->num_inputs >= 4 && node->input_names[3][0] != '\0') {
                onnx_tensor_info_t* axi = find_tensor(pm, node->input_names[3]);
                if (axi && axi->raw_data) {
                    int64_t* av = (int64_t*)axi->raw_data;
                    int n = (int)(axi->raw_data_size / sizeof(int64_t));
                    for (int i = 0; i < n && i < 8; i++) slice_axes[i] = (int)av[i];
                    num_axes = n;
                }
            }
            /* Read steps from input[4] initializer (optional) */
            if (node->num_inputs >= 5 && node->input_names[4][0] != '\0') {
                onnx_tensor_info_t* stpi = find_tensor(pm, node->input_names[4]);
                if (stpi && stpi->raw_data) {
                    int64_t* sv2 = (int64_t*)stpi->raw_data;
                    for (int d = 0; d < ndim && d < (int)(stpi->raw_data_size / sizeof(int64_t)); d++)
                        slp.steps[d] = sv2[d];
                }
            }
            /* If axes specified, remap starts/ends/steps to full-dim arrays */
            if (num_axes > 0) {
                int64_t full_starts[8] = {0};
                int64_t full_ends[8];
                int64_t full_steps[8] = {1,1,1,1,1,1,1,1};
                for (int d = 0; d < ndim; d++) full_ends[d] = slp.in_shape[d];
                for (int i = 0; i < num_axes && i < 8; i++) {
                    int ax = slice_axes[i];
                    if (ax >= 0 && ax < ndim) {
                        full_starts[ax] = slp.starts[i];
                        full_ends[ax] = ends[i];
                        full_steps[ax] = slp.steps[i];
                    }
                }
                for (int d = 0; d < ndim; d++) {
                    slp.starts[d] = full_starts[d];
                    ends[d] = full_ends[d];
                    slp.steps[d] = full_steps[d];
                }
            }
            /* Clamp and compute output shape + numel */
            for (int d = 0; d < ndim; d++) {
                if (slp.starts[d] < 0) slp.starts[d] += slp.in_shape[d];
                if (ends[d] < 0) ends[d] += slp.in_shape[d];
                if (slp.starts[d] < 0) slp.starts[d] = 0;
                if (ends[d] > slp.in_shape[d]) ends[d] = slp.in_shape[d];
                slp.out_shape[d] = (ends[d] - slp.starts[d] + slp.steps[d] - 1) / slp.steps[d];
                if (slp.out_shape[d] < 0) slp.out_shape[d] = 0;
            }
            slp.numel = 1;
            for (int d = 0; d < ndim; d++) slp.numel *= slp.out_shape[d];
            params = &slp; params_size = sizeof(slp);
        } else if (ot == OP_SPLIT) {
            split_params_t spp;
            memset(&spp, 0, sizeof(spp));
            onnx_tensor_info_t* in_sp = find_tensor(pm, node->input_names[0]);
            spp.ndim = in_sp ? in_sp->ndim : 0;
            for (int d = 0; d < spp.ndim && d < 8; d++)
                spp.in_shape[d] = in_sp->shape[d];
            int64_t axis = node_attr_int(node, "axis", 0);
            if (axis < 0) axis = spp.ndim + axis;
            spp.axis = (int)axis;
            spp.num_outputs = node->num_outputs;
            /* Read split sizes from input[1] initializer */
            if (node->num_inputs >= 2 && node->input_names[1][0] != '\0') {
                onnx_tensor_info_t* spi = find_tensor(pm, node->input_names[1]);
                if (spi && spi->raw_data) {
                    int64_t* sv = (int64_t*)spi->raw_data;
                    int n = (int)(spi->raw_data_size / sizeof(int64_t));
                    for (int i = 0; i < n && i < 8; i++) spp.splits[i] = sv[i];
                }
            }
            /* If no split sizes, divide equally */
            if (spp.splits[0] == 0) {
                int64_t eq = spp.in_shape[axis] / spp.num_outputs;
                for (int i = 0; i < spp.num_outputs; i++) spp.splits[i] = eq;
            }
            /* Compute offsets and output numels */
            int64_t outer = 1;
            for (int d = 0; d < spp.axis; d++) outer *= spp.in_shape[d];
            int64_t inner = 1;
            for (int d = spp.axis + 1; d < spp.ndim; d++) inner *= spp.in_shape[d];
            int64_t cumulative = 0;
            for (int i = 0; i < spp.num_outputs; i++) {
                spp.offsets[i] = cumulative;
                spp.out_numel[i] = outer * spp.splits[i] * inner;
                cumulative += spp.splits[i] * inner;
            }
            params = &spp; params_size = sizeof(spp);
        } else if (ot == OP_LAYERNORM) {
            layernorm_params_t lnp;
            memset(&lnp, 0, sizeof(lnp));
            int64_t axis = node_attr_int(node, "axis", -1);
            lnp.epsilon = node_attr_float(node, "epsilon", 1e-5f);
            onnx_tensor_info_t* in_ln = find_tensor(pm, node->input_names[0]);
            if (in_ln) {
                int ndim = in_ln->ndim;
                if (axis < 0) axis += ndim;
                lnp.N = 1;
                for (int d = 0; d < axis; d++) lnp.N *= in_ln->shape[d];
                lnp.normalized_size = 1;
                for (int d = (int)axis; d < ndim; d++) lnp.normalized_size *= in_ln->shape[d];
            }
            params = &lnp; params_size = sizeof(lnp);
        } else if (ot == OP_GATHER) {
            onnx_tensor_info_t* in_g = find_tensor(pm, node->input_names[0]);
            onnx_tensor_info_t* idx_g = find_tensor(pm, node->input_names[1]);
            int64_t axis = node_attr_int(node, "axis", 0);
            int ndim = in_g ? in_g->ndim : 1;
            if (axis < 0) axis += ndim;
            gather_params_t gp;
            memset(&gp, 0, sizeof(gp));
            gp.axis = axis;
            gp.num_indices = idx_g ? 1 : 1;
            if (idx_g) { for (int d = 0; d < idx_g->ndim; d++) gp.num_indices *= idx_g->shape[d]; }
            gp.block_size = 1;
            if (in_g) { for (int d = (int)axis + 1; d < in_g->ndim; d++) gp.block_size *= in_g->shape[d]; }
            gp.outer_size = 1;
            if (in_g) { for (int d = 0; d < axis; d++) gp.outer_size *= in_g->shape[d]; }
            gp.inner_size = 1;
            if (in_g) { for (int d = (int)axis; d < in_g->ndim; d++) gp.inner_size *= in_g->shape[d]; }
            params = &gp; params_size = sizeof(gp);
        } else if (ot == OP_SQUEEZE_UNSQUEEZE) {
            squeeze_unsqueeze_params_t sup;
            memset(&sup, 0, sizeof(sup));
            onnx_tensor_info_t* in_sq = find_tensor(pm, node->input_names[0]);
            if (in_sq) { sup.numel = 1; for (int d = 0; d < in_sq->ndim; d++) sup.numel *= in_sq->shape[d]; }
            params = &sup; params_size = sizeof(sup);
        } else if (ot == OP_REDUCE) {
            reduce_params_t rp;
            memset(&rp, 0, sizeof(rp));
            /* Determine op type from ONNX op name */
            rp.op = (strcmp(node->op_type, "ReduceMax") == 0) ? REDUCE_MAX : REDUCE_SUM;
            onnx_tensor_info_t* in_r = find_tensor(pm, node->input_names[0]);
            int64_t axes[8];
            int num_axes = node_attr_ints(node, "axes", axes, 8);
            if (num_axes == 0 && node->num_inputs >= 2 && node->input_names[1][0] != '\0') {
                onnx_tensor_info_t* ax_info = find_tensor(pm, node->input_names[1]);
                if (ax_info && ax_info->raw_data) {
                    int64_t* av = (int64_t*)ax_info->raw_data;
                    num_axes = (int)(ax_info->raw_data_size / sizeof(int64_t));
                    for (int i = 0; i < num_axes && i < 8; i++) axes[i] = av[i];
                }
            }
            /* Normalize negative axes */
            for (int i = 0; i < num_axes; i++)
                if (axes[i] < 0 && in_r) axes[i] += in_r->ndim;
            /* Compute reduce_size (product of reduced dims) and num_blocks (product of others) */
            int reduced[8] = {0};
            for (int i = 0; i < num_axes && i < 8; i++)
                if (axes[i] >= 0 && axes[i] < 8) reduced[axes[i]] = 1;
            if (in_r) {
                rp.reduce_size = 1;
                rp.num_blocks  = 1;
                rp.total_elems = 1;
                for (int d = 0; d < in_r->ndim; d++) {
                    rp.total_elems *= in_r->shape[d];
                    if (reduced[d]) rp.reduce_size *= in_r->shape[d];
                    else            rp.num_blocks  *= in_r->shape[d];
                }
            }
            params = &rp; params_size = sizeof(rp);
        } else if (ot == OP_CAST) {
            cast_params_t cp;
            memset(&cp, 0, sizeof(cp));
            int64_t to_type = node_attr_int(node, "to", ONNX_DTYPE_FLOAT);
            cp.dst_dtype = (int)to_type;
            onnx_tensor_info_t* in_c = find_tensor(pm, node->input_names[0]);
            /* Infer src_dtype from input tensor; default to INT64 for weights,
               FLOAT otherwise (since we store everything as F32) */
            cp.src_dtype = ONNX_DTYPE_FLOAT;
            if (in_c && in_c->is_initializer && in_c->raw_data) {
                /* Check if the raw_data looks like int64 */
                cp.src_dtype = ONNX_DTYPE_INT64;
            }
            cp.numel = 1;
            if (in_c) for (int d = 0; d < in_c->ndim; d++) cp.numel *= in_c->shape[d];
            params = &cp; params_size = sizeof(cp);
        } else if (ot == OP_ARGMAX) {
            argmax_params_t ap;
            memset(&ap, 0, sizeof(ap));
            int64_t axis = node_attr_int(node, "axis", 0);
            onnx_tensor_info_t* in_am = find_tensor(pm, node->input_names[0]);
            if (in_am) {
                if (axis < 0) axis += in_am->ndim;
                ap.reduce_size = (axis < in_am->ndim) ? in_am->shape[axis] : 1;
                ap.num_blocks = 1;
                for (int d = 0; d < in_am->ndim; d++)
                    if (d != axis) ap.num_blocks *= in_am->shape[d];
            }
            params = &ap; params_size = sizeof(ap);
        }

        /* Add the node. Data inputs exclude weight initializers; weights are passed separately. */
        graph_add_node(g, ot, num_data_in, in_tids, num_out, out_tids,
                       num_weights, weights, params, params_size);

    }

    /* Build topological order */
    if (graph_build(g) != 0) {
        fprintf(stderr, "ERROR: graph_build failed (cycle detected?)\n");
        graph_destroy(g);
        return NULL;
    }

    return g;
}

/* ============================================================
 * Main pipeline: ONNX file → onnx_model_t
 * ============================================================ */
onnx_model_t* onnx_load_from_file(const char* path) {
    if (!path) return NULL;

    /* Extract model directory for resolving external data paths */
    {
        const char* last_sep = NULL;
        const char* p = path;
        while (*p) {
            if (*p == '/' || *p == '\\') last_sep = p;
            p++;
        }
        if (last_sep) {
            size_t dir_len = (size_t)(last_sep - path + 1);
            if (dir_len >= sizeof(g_model_dir)) dir_len = sizeof(g_model_dir) - 1;
            memcpy(g_model_dir, path, dir_len);
            g_model_dir[dir_len] = '\0';
        } else {
            g_model_dir[0] = '\0';
        }
    }

    /* Read entire file */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "onnx_load: cannot open '%s'\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) { fclose(fp); return NULL; }

    uint8_t* data = (uint8_t*)malloc((size_t)fsize);
    if (!data) { fclose(fp); return NULL; }
    size_t read_n = fread(data, 1, (size_t)fsize, fp);
    fclose(fp);
    if (read_n != (size_t)fsize) { free(data); return NULL; }

    /* Parse root ModelProto */
    pb_message_t* model_msg = pb_parse_message(data, fsize, 5);
    if (!model_msg) {
        free(data);
        fprintf(stderr, "onnx_load: failed to parse protobuf\n");
        return NULL;
    }

    /* Extract GraphProto (field 7) */
    pb_field_t* graph_field = pb_find_field(model_msg, F_ModelProto_graph);
    if (!graph_field) {
        fprintf(stderr, "onnx_load: no graph in model\n");
        pb_message_destroy(model_msg);
        free(data);
        return NULL;
    }
    pb_message_t* graph_msg = pb_field_as_message(graph_field, 4);

    /* Parse opset_import (field 8): repeated OperatorSetIdProto,
       each with version (field 2). Use the default domain ("") entry. */
    g_model_opset = 11;
    {
        pb_field_t** opsets = NULL;
        int oc = 0;
        pb_find_all_fields(model_msg, F_ModelProto_opset_import, &opsets, &oc);
        for (int i = 0; i < oc; i++) {
            pb_message_t* osm = pb_field_as_message(opsets[i], 2);
            if (osm) {
                /* domain (field 1) is empty string for default ONNX opset */
                size_t dlen = 0;
                const uint8_t* dom = pb_field_get_string(osm, 1, &dlen);
                int64_t ver = pb_field_get_int64(osm, F_OperatorSetIdProto_version, 0);
                if ((!dom || dlen == 0) && ver > 0) {
                    g_model_opset = ver;
                }
                pb_message_destroy(osm);
            }
        }
        free(opsets);
    }

    pb_message_destroy(model_msg);
    if (!graph_msg) { free(data); return NULL; }

    onnx_parsed_model_t* pm = (onnx_parsed_model_t*)calloc(1, sizeof(onnx_parsed_model_t));
    if (!pm) { pb_message_destroy(graph_msg); free(data); return NULL; }
    for (int i = 0; i < TENSOR_HT_SIZE; i++) pm->tensor_ht[i] = -1;

    /* Parse initializers (field 5 of GraphProto).
       TensorProto.name is field 8 in the ONNX protobuf schema. */
    {
        pb_field_t** inits = NULL;
        int ic = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_initializer, &inits, &ic);
        for (int i = 0; i < ic && pm->num_tensors < MAX_TENSOR_INFOS; i++) {
            pb_message_t* im = pb_field_as_message(inits[i], 3);
            if (!im) continue;

            /* Try to extract name from TensorProto.name (field 8) */
            size_t nlen = 0;
            const uint8_t* nm = pb_field_get_string(im, 8, &nlen);
            char tname[MAX_TENSOR_NAME];
            if (nm && nlen > 0) {
                int nc = (int)(nlen < MAX_TENSOR_NAME - 1 ? nlen : MAX_TENSOR_NAME - 1);
                memcpy(tname, nm, nc);
                tname[nc] = '\0';
            } else {
                /* No name — use positional naming: "init_N" */
                snprintf(tname, sizeof(tname), "_init_%d", i);
            }

            onnx_tensor_info_t* ti = find_tensor(pm, tname);
            if (!ti) ti = add_tensor(pm, tname);
            if (ti) {
                parse_tensor_proto(im, ti);
                ti->is_initializer = 1;
            }
            pb_message_destroy(im);
        }
        free(inits);
    }

    /* Parse graph inputs (field 11) */
    {
        pb_field_t** gins = NULL;
        int gc = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_input, &gins, &gc);
        for (int i = 0; i < gc && pm->num_graph_inputs < 16; i++) {
            pb_message_t* vim = pb_field_as_message(gins[i], 3);
            if (!vim) continue;
            char nm[MAX_TENSOR_NAME];
            int64_t sh[8] = {0};
            int nd = 0;
            if (parse_value_info_shape(vim, nm, sizeof(nm), sh, &nd) == 0) {
                strncpy(pm->graph_input_names[pm->num_graph_inputs], nm,
                        MAX_TENSOR_NAME - 1);
                pm->num_graph_inputs++;
                onnx_tensor_info_t* ti = find_tensor(pm, nm);
                if (!ti) ti = add_tensor(pm, nm);
                if (ti) {
                    ti->ndim = nd;
                    for (int d = 0; d < nd; d++) ti->shape[d] = sh[d];
                }
            }
            pb_message_destroy(vim);
        }
        free(gins);
    }

    /* Parse graph outputs (field 12) */
    {
        pb_field_t** gouts = NULL;
        int gc = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_output, &gouts, &gc);
        for (int i = 0; i < gc && pm->num_graph_outputs < 16; i++) {
            pb_message_t* vim = pb_field_as_message(gouts[i], 3);
            if (!vim) continue;
            char nm[MAX_TENSOR_NAME];
            int64_t sh[8] = {0};
            int nd = 0;
            if (parse_value_info_shape(vim, nm, sizeof(nm), sh, &nd) == 0) {
                strncpy(pm->graph_output_names[pm->num_graph_outputs], nm,
                        MAX_TENSOR_NAME - 1);
                pm->num_graph_outputs++;
                onnx_tensor_info_t* ti = find_tensor(pm, nm);
                if (!ti) ti = add_tensor(pm, nm);
                if (ti) {
                    ti->ndim = nd;
                    for (int d = 0; d < nd; d++) ti->shape[d] = sh[d];
                }
            }
            pb_message_destroy(vim);
        }
        free(gouts);
    }

    /* Parse value_info (field 13) for intermediate shapes */
    {
        pb_field_t** vis = NULL;
        int vc = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_value_info, &vis, &vc);
        for (int i = 0; i < vc; i++) {
            pb_message_t* vim = pb_field_as_message(vis[i], 3);
            if (!vim) continue;
            char nm[MAX_TENSOR_NAME];
            int64_t sh[8] = {0};
            int nd = 0;
            if (parse_value_info_shape(vim, nm, sizeof(nm), sh, &nd) == 0) {
                onnx_tensor_info_t* ti = find_tensor(pm, nm);
                if (!ti) ti = add_tensor(pm, nm);
                if (ti) {
                    ti->ndim = nd;
                    for (int d = 0; d < nd; d++) ti->shape[d] = sh[d];
                }
            }
            pb_message_destroy(vim);
        }
        free(vis);
    }

    /* Parse nodes (field 1 of GraphProto).
       Store node messages to keep attribute pointers alive through shape inference. */
    pb_message_t* node_msgs[MAX_NODES];
    int num_node_msgs = 0;
    {
        pb_field_t** node_fields = NULL;
        int nc = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_node, &node_fields, &nc);
        for (int i = 0; i < nc && pm->num_nodes < MAX_NODES; i++) {
            pb_message_t* nm = pb_field_as_message(node_fields[i], 3);
            if (!nm) continue;
            parse_node_proto(nm, &pm->nodes[pm->num_nodes++]);
            node_msgs[num_node_msgs++] = nm;  /* keep alive, destroy later */
        }
        free(node_fields);
    }
    pb_message_destroy(graph_msg);
    /* NOTE: do NOT free(data) yet — all protobuf length_delimited.data pointers
       (including node attributes) trace back to data. We free it after graph building. */

    /* Shape inference: process nodes in declaration order.
       For models with topological node order (standard ONNX), this works.
       For non-topological order, we'd need dependency analysis. */
    for (int i = 0; i < pm->num_nodes; i++) {
        infer_output_shape(&pm->nodes[i], pm);
    }

    /* Second pass: ensure all tensor edges have shapes (fill gaps).
       Only apply to elementwise ops where output shape = input shape.
       Ops like ArgMax/Reduce/Cast can legitimately produce 0D outputs. */
    for (int ni = 0; ni < pm->num_nodes; ni++) {
        onnx_node_info_t* node = &pm->nodes[ni];
        op_type_t ot = map_onnx_op(node->op_type);
        int is_elem = (ot == OP_RELU || ot == OP_SIGMOID || ot == OP_GELU
                       || ot == OP_SILU || ot == OP_BATCHNORM || ot == OP_ADD
                       || ot == OP_MUL || ot == OP_SOFTMAX || ot == OP_SUB
                       || ot == OP_DIV || ot == OP_EXP || ot == OP_CAST);
        if (!is_elem) continue;
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* t = find_tensor(pm, node->output_names[oi]);
            if (t && t->ndim == 0) {
                onnx_tensor_info_t* in0 = find_tensor(pm, node->input_names[0]);
                if (in0 && in0->ndim > 0) {
                    t->ndim = in0->ndim;
                    for (int d = 0; d < in0->ndim; d++) t->shape[d] = in0->shape[d];
                }
            }
        }
    }

    /* Build inference graph */
    inference_graph_t* g = build_graph_from_onnx(pm);

    /* Now safe to destroy node messages and data buffer
       (all protobuf data is no longer needed after graph construction) */
    for (int i = 0; i < num_node_msgs; i++) {
        pb_message_destroy(node_msgs[i]);
    }
    free(data);
    if (!g) {
        /* Clean up parsed model tensor data */
        for (int i = 0; i < pm->num_tensors; i++) {
            free(pm->tensors[i].raw_data);
        }
        free(pm);
        return NULL;
    }

    /* Build result */
    onnx_model_t* model = (onnx_model_t*)calloc(1, sizeof(onnx_model_t));
    if (!model) {
        graph_destroy(g);
        for (int i = 0; i < pm->num_tensors; i++) free(pm->tensors[i].raw_data);
        free(pm);
        return NULL;
    }

    model->graph = g;
    model->num_inputs = pm->num_graph_inputs;
    model->num_outputs = pm->num_graph_outputs;

    model->input_names = (char**)malloc((size_t)pm->num_graph_inputs * sizeof(char*));
    model->input_shapes = (int64_t**)malloc((size_t)pm->num_graph_inputs * sizeof(int64_t*));
    model->input_ndims  = (int*)malloc((size_t)pm->num_graph_inputs * sizeof(int));
    for (int i = 0; i < pm->num_graph_inputs; i++) {
        model->input_names[i] = dup_str(pm->graph_input_names[i]);
        onnx_tensor_info_t* ti = find_tensor(pm, pm->graph_input_names[i]);
        if (ti && ti->ndim > 0) {
            model->input_ndims[i] = ti->ndim;
            model->input_shapes[i] = (int64_t*)malloc((size_t)ti->ndim * sizeof(int64_t));
            for (int d = 0; d < ti->ndim; d++)
                model->input_shapes[i][d] = ti->shape[d];
        } else {
            model->input_ndims[i] = 0;
            model->input_shapes[i] = NULL;
        }
    }

    model->output_names = (char**)malloc((size_t)pm->num_graph_outputs * sizeof(char*));
    for (int i = 0; i < pm->num_graph_outputs; i++) {
        model->output_names[i] = dup_str(pm->graph_output_names[i]);
    }

    /* Clean up parsed model */
    for (int i = 0; i < pm->num_tensors; i++) free(pm->tensors[i].raw_data);
    free(pm);

    return model;
}

void onnx_model_destroy(onnx_model_t* model) {
    if (!model) return;
    graph_destroy(model->graph);
    for (int i = 0; i < model->num_inputs; i++) {
        free(model->input_names[i]);
        free(model->input_shapes[i]);
    }
    free(model->input_names);
    free(model->input_shapes);
    free(model->input_ndims);
    for (int i = 0; i < model->num_outputs; i++) {
        free(model->output_names[i]);
    }
    free(model->output_names);
    free(model);
}
