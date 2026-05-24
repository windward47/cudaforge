"""Generate minimal ONNX protobuf binary files for testing.

Encodes protobuf wire format directly (no onnx package needed).
Generates:
  1. test_relu.onnx   — Input(1x4) → ReLU → Output(1x4)
  2. test_conv.onnx   — Input(1x1x3x3) → Conv(1x1x2x2) → ReLU → Output
  3. test_chain.onnx  — Input(1x4) → ReLU → ReLU → Output

Also prints C byte arrays for embedding in test code.
"""

import struct
import sys
import os

# ============================================================
# Protobuf encoding helpers
# ============================================================

def encode_varint(value: int) -> bytes:
    """Encode unsigned integer as LEB128 varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def encode_tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def encode_varint_field(fn: int, value: int) -> bytes:
    return encode_tag(fn, 0) + encode_varint(value)


def encode_length_delimited(fn: int, data: bytes) -> bytes:
    return encode_tag(fn, 2) + encode_varint(len(data)) + data


def encode_string_field(fn: int, s: str) -> bytes:
    return encode_length_delimited(fn, s.encode('utf-8'))


def encode_int32_field(fn: int, v: int) -> bytes:
    return encode_varint_field(fn, v)


def encode_int64_field(fn: int, v: int) -> bytes:
    return encode_varint_field(fn, v)


def encode_float_field(fn: int, v: float) -> bytes:
    b = struct.pack('<f', v)
    return encode_tag(fn, 5) + b


def encode_bool_field(fn: int, v: bool) -> bytes:
    return encode_varint_field(fn, 1 if v else 0)


def encode_packed_varints(fn: int, values: list[int]) -> bytes:
    data = b''.join(encode_varint(v) for v in values)
    return encode_length_delimited(fn, data)


def encode_packed_floats(fn: int, values: list[float]) -> bytes:
    data = b''.join(struct.pack('<f', v) for v in values)
    return encode_length_delimited(fn, data)


# ============================================================
# ONNX protobuf message builders
# ============================================================

# Field numbers
F_ModelProto_graph = 7

F_GraphProto_node = 1
F_GraphProto_initializer = 5
F_GraphProto_input = 11
F_GraphProto_output = 12

F_NodeProto_input = 1
F_NodeProto_output = 2
F_NodeProto_op_type = 4
F_NodeProto_attribute = 5

F_AttrProto_name = 1
F_AttrProto_ints = 8
F_AttrProto_i = 3
F_AttrProto_f = 2

F_TensorProto_dims = 1
F_TensorProto_data_type = 2
F_TensorProto_float_data = 6
F_TensorProto_raw_data = 9
F_TensorProto_name = 8  # optional

F_ValueInfoProto_name = 1
F_ValueInfoProto_type = 5

F_TypeProto_tensor_type = 1
F_TensorShapeProto_dim = 1
F_DimValue = 1

ONNX_DTYPE_FLOAT = 1


def make_dim(dim_value: int) -> bytes:
    """TensorShapeProto.Dimension"""
    inner = encode_int64_field(F_DimValue, dim_value)
    return encode_length_delimited(F_TensorShapeProto_dim, inner)


def make_tensor_shape(dims: list[int]) -> bytes:
    """TypeProto.TensorShapeProto"""
    inner = b''.join(make_dim(d) for d in dims)
    return inner


def make_type_tensor(shape_dims: list[int]) -> bytes:
    """TypeProto { tensor_type { shape { dim { dim_value } ... } } }"""
    shape = make_tensor_shape(shape_dims)
    # TypeProto.Tensor.shape is field 2
    tensor_inner = encode_length_delimited(2, shape)
    # TypeProto.tensor_type is field 1
    return encode_length_delimited(F_TypeProto_tensor_type, tensor_inner)


def make_value_info(name: str, shape_dims: list[int]) -> bytes:
    """ValueInfoProto { name, type { tensor_type { shape { dims } } } }"""
    result = encode_string_field(F_ValueInfoProto_name, name)
    result += encode_length_delimited(F_ValueInfoProto_type, make_type_tensor(shape_dims))
    return result


def make_tensor_proto(name: str, dims: list[int], float_values: list[float],
                       use_raw: bool = True) -> bytes:
    """TensorProto with name, dims, and float data."""
    result = b''
    if name:
        result += encode_string_field(F_TensorProto_name, name)
    result += encode_packed_varints(F_TensorProto_dims, dims)
    result += encode_int32_field(F_TensorProto_data_type, ONNX_DTYPE_FLOAT)
    if use_raw:
        raw = b''.join(struct.pack('<f', v) for v in float_values)
        result += encode_length_delimited(F_TensorProto_raw_data, raw)
    else:
        result += encode_packed_floats(F_TensorProto_float_data, float_values)
    return result


def make_attr_ints(name: str, values: list[int]) -> bytes:
    """AttributeProto with name and ints."""
    inner = encode_string_field(F_AttrProto_name, name)
    inner += encode_packed_varints(F_AttrProto_ints, values)
    return inner


def make_attr_int(name: str, value: int) -> bytes:
    """AttributeProto with name and single int."""
    inner = encode_string_field(F_AttrProto_name, name)
    inner += encode_int64_field(F_AttrProto_i, value)
    return inner


def make_node(op_type: str, inputs: list[str], outputs: list[str],
               attributes: list[bytes] = None) -> bytes:
    """NodeProto."""
    inner = b''
    for inp in inputs:
        inner += encode_string_field(F_NodeProto_input, inp)
    for out in outputs:
        inner += encode_string_field(F_NodeProto_output, out)
    inner += encode_string_field(F_NodeProto_op_type, op_type)
    if attributes:
        for attr in attributes:
            inner += encode_length_delimited(F_NodeProto_attribute, attr)
    return inner


def make_graph(nodes: list[bytes], inputs: list[bytes], outputs: list[bytes],
                initializers: list[bytes] = None) -> bytes:
    """GraphProto."""
    inner = b''
    for node in nodes:
        inner += encode_length_delimited(F_GraphProto_node, node)
    if initializers:
        for init in initializers:
            inner += encode_length_delimited(F_GraphProto_initializer, init)
    for inp in inputs:
        inner += encode_length_delimited(F_GraphProto_input, inp)
    for out in outputs:
        inner += encode_length_delimited(F_GraphProto_output, out)
    return inner


def make_model(graph_bytes: bytes) -> bytes:
    """ModelProto with graph."""
    inner = encode_length_delimited(F_ModelProto_graph, graph_bytes)
    return inner


# ============================================================
# Model generators
# ============================================================

def gen_relu_model() -> bytes:
    """Input(1x4) → ReLU → Output(1x4)"""
    inp = make_value_info("input", [1, 4])
    out = make_value_info("output", [1, 4])

    node = make_node("Relu", ["input"], ["output"])

    graph = make_graph([node], [inp], [out])
    return make_model(graph)


def gen_chain_model() -> bytes:
    """Input(1x4) → ReLU → ReLU → Output(1x4)"""
    inp = make_value_info("input", [1, 4])
    out = make_value_info("output", [1, 4])

    n1 = make_node("Relu", ["input"], ["hidden"])
    n2 = make_node("Relu", ["hidden"], ["output"])

    graph = make_graph([n1, n2], [inp], [out])
    return make_model(graph)


def gen_conv_model() -> bytes:
    """Input(1x1x3x3) → Conv(1x1x2x2 identity kernel) → ReLU(1x1x2x2) → Output"""
    # Identity kernel: [[1,0],[0,1]]
    # weight shape: [K=1, C=1, KH=2, KW=2]
    weight_vals = [1.0, 0.0, 0.0, 1.0]

    weight = make_tensor_proto("conv.weight", [1, 1, 2, 2], weight_vals)

    inp = make_value_info("input", [1, 1, 3, 3])
    out = make_value_info("output", [1, 1, 2, 2])

    conv_attrs = [
        make_attr_ints("kernel_shape", [2, 2]),
        make_attr_ints("strides", [1, 1]),
        make_attr_ints("pads", [0, 0, 0, 0]),
    ]

    n1 = make_node("Conv", ["input", "conv.weight"], ["conv_out"], conv_attrs)
    n2 = make_node("Relu", ["conv_out"], ["output"])

    graph = make_graph([n1, n2], [inp], [out], [weight])
    return make_model(graph)


def print_c_array(name: str, data: bytes):
    """Print data as a C byte array."""
    print(f"static const unsigned char {name}[] = {{")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_bytes = ', '.join(f'0x{b:02X}' for b in chunk)
        if i + 16 < len(data):
            print(f"    {hex_bytes},")
        else:
            print(f"    {hex_bytes}")
    print(f"}};")
    print(f"static const size_t {name}_len = {len(data)};")
    print()


# ============================================================
# Main
# ============================================================

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    models = {
        "test_relu": gen_relu_model(),
        "test_chain": gen_chain_model(),
        "test_conv": gen_conv_model(),
    }

    for name, data in models.items():
        path = os.path.join(output_dir, f"{name}.onnx")
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Written {path} ({len(data)} bytes)", file=sys.stderr)

    # Print C embeddable arrays for the ReLU model (simplest)
    print("\n/* === C embeddable ONNX model bytes === */\n")
    for name, data in models.items():
        print(f"/* {name}.onnx ({len(data)} bytes) */")
        print_c_array(f"onnx_{name}_data", data)

    # Print expected test input/output
    print("/*")
    print("Test cases:")
    print("  relu:   input=[-1,0,2,-3]  -> expected=[0,0,2,0]")
    print("  chain:  input=[-1,0,2,-3]  -> expected=[0,0,2,0]  (ReLU(ReLU(x)) == ReLU(x))")
    print("  conv:   input 3x3 = [1,2,3, 4,5,6, 7,8,9]")
    print("          with identity kernel 2x2")
    print("          conv output = [1+5, 2+6, 4+8, 5+9] = [6,8,12,14]")
    print("          relu output = [6,8,12,14] (all positive)")
    print("*/")


if __name__ == '__main__':
    main()
