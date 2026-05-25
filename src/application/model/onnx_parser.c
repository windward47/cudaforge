#include "onnx_parser.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define PB_MAX_DEPTH   16
#define PB_INIT_CAP    16

/* ============================================================
 * Varint (LEB128) decode
 * ============================================================ */
static int pb_read_varint(const uint8_t** ptr, const uint8_t* end,
                           uint64_t* out) {
    const uint8_t* p = *ptr;
    uint64_t val = 0;
    int shift = 0;
    while (p < end) {
        uint8_t byte = *p++;
        val |= ((uint64_t)(byte & 0x7F)) << shift;
        if (!(byte & 0x80)) {
            *ptr = p;
            *out = val;
            return 0;
        }
        shift += 7;
        if (shift >= 64) return -1; /* too long */
    }
    return -1; /* truncated */
}

/* ============================================================
 * Fixed64 (8 bytes little-endian → double)
 * ============================================================ */
static double pb_read_fixed64(const uint8_t* data) {
    union { uint64_t u; double d; } v;
    v.u = (uint64_t)data[0]       | ((uint64_t)data[1] << 8)  |
          ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
          ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
          ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
    return v.d;
}

/* ============================================================
 * Fixed32 (4 bytes little-endian → float)
 * ============================================================ */
static float pb_read_fixed32(const uint8_t* data) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)data[0]       | ((uint32_t)data[1] << 8) |
          ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
    return v.f;
}

/* ============================================================
 * Message parser
 * ============================================================ */
pb_message_t* pb_parse_message(const uint8_t* data, size_t size, int max_depth) {
    if (max_depth <= 0) return NULL;
    if (!data && size > 0) return NULL;

    pb_message_t* msg = (pb_message_t*)calloc(1, sizeof(pb_message_t));
    if (!msg) return NULL;
    for (int i = 0; i < PB_MAX_FIELD_NUM; i++) msg->first_by_fn[i] = -1;

    const uint8_t* p   = data;
    const uint8_t* end = data + size;

    while (p < end) {
        uint64_t tag = 0;
        if (pb_read_varint(&p, end, &tag) != 0) break;

        int32_t fn = (int32_t)(tag >> 3);
        int wtype  = (int)(tag & 0x07);

        if (fn == 0) break; /* proto3: field 0 is illegal / sentinel */

        pb_field_t f;
        memset(&f, 0, sizeof(f));
        f.field_number = fn;

        if (wtype == 0) {
            /* Varint */
            f.wire_type = PB_WIRE_VARINT;
            if (pb_read_varint(&p, end, &f.varint_value) != 0) break;
        } else if (wtype == 1) {
            /* 64-bit fixed (stored as double via Fixed64) */
            if (p + 8 > end) break;
            f.wire_type = PB_WIRE_VARINT;
            f.varint_value = 0; /* store double interpretation when needed */
            f.length_delimited.data = (uint8_t*)p;
            f.length_delimited.size = 8;
            p += 8;
        } else if (wtype == 2) {
            /* Length-delimited */
            uint64_t len = 0;
            if (pb_read_varint(&p, end, &len) != 0) break;
            if (p + len > end) break;
            f.wire_type = PB_WIRE_LENGTH_DELIMITED;
            f.length_delimited.data  = (uint8_t*)p;
            f.length_delimited.size  = (size_t)len;
            p += (size_t)len;
        } else if (wtype == 5) {
            /* 32-bit fixed — store in length_delimited so pb_field_get_float can decode it */
            if (p + 4 > end) break;
            f.wire_type = PB_WIRE_LENGTH_DELIMITED;
            f.varint_value = 0;
            f.length_delimited.data = (uint8_t*)p;
            f.length_delimited.size = 4;
            p += 4;
        } else {
            /* Unknown wire type — skip to avoid infinite loop */
            break;
        }

        /* Grow fields array if needed */
        if (msg->count >= msg->capacity) {
            int new_cap = msg->capacity ? msg->capacity * 2 : PB_INIT_CAP;
            pb_field_t* tmp = (pb_field_t*)realloc(msg->fields,
                                                    (size_t)new_cap * sizeof(pb_field_t));
            if (!tmp) {
                pb_message_destroy(msg);
                return NULL;
            }
            msg->fields   = tmp;
            msg->capacity = new_cap;
        }
        /* Record first occurrence for O(1) find-by-number */
        if (fn < PB_MAX_FIELD_NUM && msg->first_by_fn[fn] < 0)
            msg->first_by_fn[fn] = msg->count;
        msg->fields[msg->count++] = f;
    }

    return msg;
}

void pb_message_destroy(pb_message_t* msg) {
    if (!msg) return;
    free(msg->fields);
    free(msg);
}

/* ============================================================
 * Query helpers
 * ============================================================ */
pb_field_t* pb_find_field(const pb_message_t* msg, int32_t field_number) {
    if (!msg) return NULL;
    if (field_number >= 0 && field_number < PB_MAX_FIELD_NUM) {
        int idx = msg->first_by_fn[field_number];
        if (idx >= 0) return &msg->fields[idx];
        return NULL;
    }
    /* Field number out of index range — fall back to linear scan (rare) */
    for (int i = 0; i < msg->count; i++) {
        if (msg->fields[i].field_number == field_number)
            return &msg->fields[i];
    }
    return NULL;
}

int pb_find_all_fields(const pb_message_t* msg, int32_t field_number,
                        pb_field_t*** out_fields, int* out_count) {
    *out_fields = NULL;
    *out_count  = 0;
    if (!msg) return -1;

    int cnt = 0;
    for (int i = 0; i < msg->count; i++) {
        if (msg->fields[i].field_number == field_number) cnt++;
    }
    if (cnt == 0) return 0;

    pb_field_t** arr = (pb_field_t**)malloc((size_t)cnt * sizeof(pb_field_t*));
    if (!arr) return -1;

    int j = 0;
    for (int i = 0; i < msg->count; i++) {
        if (msg->fields[i].field_number == field_number)
            arr[j++] = &msg->fields[i];
    }
    *out_fields = arr;
    *out_count  = cnt;
    return 0;
}

pb_message_t* pb_field_as_message(const pb_field_t* f, int max_depth) {
    if (!f || f->wire_type != PB_WIRE_LENGTH_DELIMITED) return NULL;
    return pb_parse_message(f->length_delimited.data,
                            f->length_delimited.size, max_depth);
}

/* ============================================================
 * Typed field accessors
 * ============================================================ */
int64_t pb_field_get_int64(const pb_message_t* msg, int32_t fn, int64_t def) {
    pb_field_t* f = pb_find_field(msg, fn);
    if (!f || f->wire_type != PB_WIRE_VARINT) return def;
    return (int64_t)f->varint_value;
}

int32_t pb_field_get_int32(const pb_message_t* msg, int32_t fn, int32_t def) {
    return (int32_t)pb_field_get_int64(msg, fn, def);
}

float pb_field_get_float(const pb_message_t* msg, int32_t fn, float def) {
    pb_field_t* f = pb_find_field(msg, fn);
    if (!f) return def;
    /* Float can be encoded as wire_type 5 (32-bit fixed) */
    if (f->wire_type == PB_WIRE_LENGTH_DELIMITED && f->length_delimited.size == 4) {
        return pb_read_fixed32(f->length_delimited.data);
    }
    if (f->wire_type == PB_WIRE_VARINT) {
        /* Some encoders put float as varint, reinterpret */
        union { uint32_t u; float f; } v;
        v.u = (uint32_t)f->varint_value;
        return v.f;
    }
    return def;
}

const uint8_t* pb_field_get_string(const pb_message_t* msg, int32_t fn,
                                    size_t* out_len) {
    pb_field_t* f = pb_find_field(msg, fn);
    if (!f || f->wire_type != PB_WIRE_LENGTH_DELIMITED) {
        *out_len = 0;
        return NULL;
    }
    *out_len = f->length_delimited.size;
    return f->length_delimited.data;
}

int pb_field_get_bool(const pb_message_t* msg, int32_t fn, int def) {
    return pb_field_get_int32(msg, fn, def) != 0;
}

/* ============================================================
 * Packed decoders
 * ============================================================ */
int pb_decode_packed_varints(const uint8_t* data, size_t size,
                              int64_t* out_values, int max_values, int* out_count) {
    *out_count = 0;
    if (!data || size == 0) return 0;

    const uint8_t* p   = data;
    const uint8_t* end = data + size;
    int cnt = 0;

    while (p < end && cnt < max_values) {
        uint64_t val = 0;
        if (pb_read_varint(&p, end, &val) != 0) return -1;
        out_values[cnt++] = (int64_t)val;
    }
    *out_count = cnt;
    return 0;
}

int pb_decode_float_data(const uint8_t* data, size_t size,
                          float* out_values, int max_values, int* out_count) {
    *out_count = 0;
    if (!data || size == 0) return 0;

    int num = (int)(size / 4);
    if (num > max_values) num = max_values;
    for (int i = 0; i < num; i++) {
        out_values[i] = pb_read_fixed32(data + i * 4);
    }
    *out_count = num;
    return 0;
}
