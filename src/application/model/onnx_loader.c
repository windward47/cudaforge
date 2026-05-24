#include "onnx_loader.h"
#include "onnx_parser.h"
#include "operator.h"
#include "matmul_int.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "batchnorm_int.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================
 * ONNX protobuf field numbers (subset)
 * ============================================================ */
enum {
    F_ModelProto_graph          = 7,
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
    F_ValueInfoProto_name       = 1,
    F_ValueInfoProto_type       = 5,
    F_TypeProto_tensor_type     = 1,
    F_TensorShapeProto_dim      = 1,
    F_DimValue                  = 1,
};

#define ONNX_DTYPE_FLOAT  1
#define ONNX_DTYPE_INT64  7
#define MAX_TENSOR_NAME   128
#define MAX_NODES         128
#define MAX_TENSOR_INFOS  256

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

/* ============================================================
 * Internal: parsed ONNX model before graph construction
 * ============================================================ */
typedef struct {
    onnx_tensor_info_t tensors[MAX_TENSOR_INFOS];
    int                num_tensors;
    onnx_node_info_t   nodes[MAX_NODES];
    int                num_nodes;
    char               graph_input_names[16][MAX_TENSOR_NAME];
    int                num_graph_inputs;
    char               graph_output_names[16][MAX_TENSOR_NAME];
    int                num_graph_outputs;
} onnx_parsed_model_t;

/* ============================================================
 * Helper: find tensor by name
 * ============================================================ */
static onnx_tensor_info_t* find_tensor(onnx_parsed_model_t* m, const char* name) {
    for (int i = 0; i < m->num_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0)
            return &m->tensors[i];
    }
    return NULL;
}

static onnx_tensor_info_t* add_tensor(onnx_parsed_model_t* m, const char* name) {
    if (m->num_tensors >= MAX_TENSOR_INFOS) return NULL;
    onnx_tensor_info_t* t = &m->tensors[m->num_tensors++];
    memset(t, 0, sizeof(*t));
    { size_t nl = strlen(name); if (nl >= MAX_TENSOR_NAME) nl = MAX_TENSOR_NAME - 1;
      memcpy(t->name, name, nl); t->name[nl] = '\0'; }
    t->tensor_id = -1;
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

    /* float_data: proto3 uses field 4, proto2 uses field 6 */
    pb_field_t* fd = pb_find_field(msg, 4);  /* proto3 */
    if (!fd || fd->wire_type != PB_WIRE_LENGTH_DELIMITED)
        fd = pb_find_field(msg, F_TensorProto_float_data);  /* proto2: field 6 */
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
    if (ot == OP_RELU || ot == OP_SIGMOID || ot == OP_GELU || ot == OP_BATCHNORM) {
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
        int64_t M = in0->ndim >= 2 ? in0->shape[0] : 1;
        int64_t N = transB ? in1->shape[0] : (in1->ndim >= 2 ? in1->shape[1] : 1);

        int64_t out_shape[2] = {M, N};
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* out = find_tensor(model, node->output_names[oi]);
            if (!out) out = add_tensor(model, node->output_names[oi]);
            if (out) {
                out->ndim = 2;
                out->shape[0] = out_shape[0];
                out->shape[1] = out_shape[1];
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
            if (t->tensor_id < 0) {
                tensor_t* tt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
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
            if (t && t->is_initializer && t->raw_data) {
                tensor_t* wt = tensor_create(DATA_TYPE_F32, t->ndim, t->shape);
                if (wt && wt->data && t->raw_data_size <= (size_t)wt->numel * sizeof(float)) {
                    memcpy(wt->data, t->raw_data, t->raw_data_size);
                }
                weights[num_weights++] = wt;
            } else {
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
            static conv_params_t cp;
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
            static pool_params_t pp;
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
            static matmul_params_t mp;
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
            static batchnorm_params_t bp;
            memset(&bp, 0, sizeof(bp));
            onnx_tensor_info_t* in_t = find_tensor(pm, node->input_names[0]);
            if (in_t && in_t->ndim >= 2) {
                bp.C = in_t->shape[1];
            }
            bp.epsilon = node_attr_float(node, "epsilon", 1e-5f);
            params = &bp; params_size = sizeof(bp);
        }

        /* Add the node. Data inputs exclude weight initializers; weights are passed separately. */
        int nid = graph_add_node(g, ot, num_data_in, in_tids, num_out, out_tids,
                                  num_weights, weights, params, params_size);

        /* For activations with no params, mark the node correctly */
        if (ot == OP_RELU || ot == OP_SIGMOID || ot == OP_GELU) {
            /* Only input[0] is a data input; num_in should be 1 */
            /* The graph_add_node already has correct num_in */
            (void)nid;
        }
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
    free(data);
    if (!model_msg) {
        fprintf(stderr, "onnx_load: failed to parse protobuf\n");
        return NULL;
    }

    /* Extract GraphProto (field 7) */
    pb_field_t* graph_field = pb_find_field(model_msg, F_ModelProto_graph);
    if (!graph_field) {
        fprintf(stderr, "onnx_load: no graph in model\n");
        pb_message_destroy(model_msg);
        return NULL;
    }
    pb_message_t* graph_msg = pb_field_as_message(graph_field, 4);
    pb_message_destroy(model_msg);
    if (!graph_msg) return NULL;

    onnx_parsed_model_t* pm = (onnx_parsed_model_t*)calloc(1, sizeof(onnx_parsed_model_t));
    if (!pm) { pb_message_destroy(graph_msg); return NULL; }

    /* Parse initializers (field 5 of GraphProto) */
    {
        pb_field_t** inits = NULL;
        int ic = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_initializer, &inits, &ic);
        for (int i = 0; i < ic && pm->num_tensors < MAX_TENSOR_INFOS; i++) {
            pb_message_t* im = pb_field_as_message(inits[i], 3);
            if (!im) continue;
            onnx_tensor_info_t* ti = add_tensor(pm, "");
            if (ti) {
                parse_tensor_proto(im, ti);
                ti->is_initializer = 1;
                /* Name is not explicitly in TensorProto for initializers in all ONNX versions.
                   We need to figure out names from the graph context. For now, initializers
                   are referenced by name. The name is in field 1 of the outer field or we
                   need to check the raw message. */
            }
            pb_message_destroy(im);
        }
        free(inits);

        /* Re-parse initializers with names. ONNX initializers have name in field 1 of
           the ValueInfoProto that wraps them, or the TensorProto has no name.
           Actually: in GraphProto, "initializer" is a repeated TensorProto,
           and TensorProto has NO name field in standard ONNX (the name comes from
           the key in the map, but protobuf map is just repeated).

           The standard way: ONNX uses repeated TensorProto without name in some versions,
           or with name in others. We need to handle both.

           Re-approach: parse again looking for name. */
    }

    /* Actually, ONNX GraphProto.initializer are TensorProto messages which
       don't have a 'name' field. The names come from the node's input references.
       We need to match initializers to node inputs by position.
       Let's store initializers indexed by order and match later.

       But wait: the TensorProto can have a name in some ONNX versions. Let me
       check field 1 ... Actually in onnx.proto, TensorProto doesn't have a name field.
       The initializer names come from the repeated field key in the map.

       Simplification: parse initializers in order, and in node parsing, when a node's
       input name matches an initializer name that we track separately, connect them.

       Actually the simplest approach: Parse initializer names from the node inputs.
       An initializer is identified by being present in GraphProto.initializer AND
       NOT being present in GraphProto.input. So:
       1. Parse GraphProto.input → these are graph inputs (not weights)
       2. Parse GraphProto.initializer → these are weights
       3. For each initializer, we know its index; the name is implicit from the
          graph structure. OR, we can check the raw protobuf.

       Actually, TensorProto DOES include name as an optional field. Let me check:
       In onnx.proto3, message TensorProto { optional string name = 1; ... }
       But field 1 might be overloaded.

       The cleanest approach: in GraphProto, initializer (field 5) entries are
       TensorProto which may have name in some encodings. If not, we match by order.

       Let me redo the initializer parsing to also try extracting a name.
       The name field in TensorProto is field 1 ... wait, dims is field 1.
       Let me check: TensorProto: dims=1, data_type=2, segment=3, float_data=4..6,
       int32_data=7, string_data=8, int64_data=9, name=10, ...

       No, wait: in ONNX protobuf:
       message TensorProto {
         repeated int64 dims = 1;
         optional int32 data_type = 2;
         ...
         optional string name = 8;  // <-- name is field 8!
         optional bytes raw_data = 9;
         ...
       }

       So TensorProto.name is field 8! Let me handle this.
    */

    /* Parse initializers with name (field 8) */
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

    /* Parse nodes (field 1 of GraphProto) */
    {
        pb_field_t** node_fields = NULL;
        int nc = 0;
        pb_find_all_fields(graph_msg, F_GraphProto_node, &node_fields, &nc);
        for (int i = 0; i < nc && pm->num_nodes < MAX_NODES; i++) {
            pb_message_t* nm = pb_field_as_message(node_fields[i], 3);
            if (!nm) continue;
            parse_node_proto(nm, &pm->nodes[pm->num_nodes++]);
            pb_message_destroy(nm);
        }
        free(node_fields);
    }
    pb_message_destroy(graph_msg);

    /* Shape inference: process nodes in declaration order.
       For models with topological node order (standard ONNX), this works.
       For non-topological order, we'd need dependency analysis. */
    for (int i = 0; i < pm->num_nodes; i++) {
        infer_output_shape(&pm->nodes[i], pm);
    }

    /* Second pass: ensure all tensor edges have shapes (fill gaps) */
    for (int ni = 0; ni < pm->num_nodes; ni++) {
        onnx_node_info_t* node = &pm->nodes[ni];
        for (int oi = 0; oi < node->num_outputs; oi++) {
            onnx_tensor_info_t* t = find_tensor(pm, node->output_names[oi]);
            if (t && t->ndim == 0) {
                /* Use input shape as fallback for elementwise ops */
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
        model->input_names[i] = _strdup(pm->graph_input_names[i]);
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
        model->output_names[i] = _strdup(pm->graph_output_names[i]);
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
