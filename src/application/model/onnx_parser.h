#ifndef ONNX_PARSER_H_
#define ONNX_PARSER_H_

#include <stddef.h>
#include <stdint.h>

/* Protobuf wire types */
typedef enum {
    PB_WIRE_VARINT           = 0,
    PB_WIRE_LENGTH_DELIMITED = 2,
    PB_WIRE_INVALID          = -1
} pb_wire_type_t;

/* A single parsed protobuf field */
typedef struct {
    int32_t         field_number;
    pb_wire_type_t  wire_type;
    uint64_t        varint_value;
    struct {
        uint8_t* data;
        size_t   size;
    } length_delimited;
} pb_field_t;

/* A message is a dynamic array of parsed fields */
#define PB_MAX_FIELD_NUM 64  /* field numbers 0..63 are indexed */

typedef struct {
    pb_field_t* fields;
    int         count;
    int         capacity;
    int         first_by_fn[PB_MAX_FIELD_NUM];  /* field# → first index, -1 = absent */
} pb_message_t;

/* ---- Lifecycle ---- */
pb_message_t* pb_parse_message(const uint8_t* data, size_t size, int max_depth);
void          pb_message_destroy(pb_message_t* msg);

/* ---- Query helpers ---- */
pb_field_t*   pb_find_field(const pb_message_t* msg, int32_t field_number);

/* Returns all fields with given field_number (for repeated fields).
   Caller must free *out_fields with free(). */
int           pb_find_all_fields(const pb_message_t* msg, int32_t field_number,
                                 pb_field_t*** out_fields, int* out_count);

/* Convert a length-delimited field to nested message */
pb_message_t* pb_field_as_message(const pb_field_t* f, int max_depth);

/* ---- Value decoders ---- */
int64_t       pb_field_get_int64(const pb_message_t* msg, int32_t fn, int64_t def);
int32_t       pb_field_get_int32(const pb_message_t* msg, int32_t fn, int32_t def);
float         pb_field_get_float(const pb_message_t* msg, int32_t fn, float def);
const uint8_t* pb_field_get_string(const pb_message_t* msg, int32_t fn,
                                   size_t* out_len);
int           pb_field_get_bool(const pb_message_t* msg, int32_t fn, int def);

/* Decode length-delimited data as packed varints (LEB128).
   Used for tensor dims, attribute int64 lists. */
int pb_decode_packed_varints(const uint8_t* data, size_t size,
                              int64_t* out_values, int max_values, int* out_count);

/* Decode repeated floats from a length-delimited field.
   ONNX float_data is packed little-endian IEEE-754 binary32. */
int pb_decode_float_data(const uint8_t* data, size_t size,
                          float* out_values, int max_values, int* out_count);

#endif /* ONNX_PARSER_H_ */
