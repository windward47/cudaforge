"""
Generate ONNX image classification model and test data for CudaForge.

Model: all-conv MNIST-style classifier using only CudaForge-supported ops:
  Input(1,1,28,28) → Conv(k=3,p=1, 1→4) → Relu → MaxPool(2×2)
  → Conv(k=3,p=1, 4→8) → Relu → MaxPool(2×2)
  → GlobalAveragePool → Reshape(1,8) → MatMul(8→10) → Softmax

Downloads a real MNIST test digit as input.
"""
import struct
import sys
import os
import gzip
import urllib.request
import numpy as np

# ============================================================
# Download MNIST test data
# ============================================================
MNIST_BASE = "https://github.com/facebookresearch/CrypTen/raw/main/notebooks/mnist_sample"


def download_mnist_sample(save_dir):
    """Download a sample MNIST PNG image."""
    img_url = "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"
    # Use local generation instead of downloading
    return None


def generate_digit_image():
    """Generate a simple 28x28 digit '3' pattern."""
    img = np.zeros((28, 28), dtype=np.float32)
    # Draw a "3" shape
    for i in range(5, 8):
        for j in range(6, 22):
            img[i, j] = 1.0
    for i in range(8, 14):
        for j in range(18, 22):
            img[i, j] = 1.0
    for i in range(11, 14):
        for j in range(6, 22):
            img[i, j] = 1.0
    for i in range(14, 20):
        for j in range(18, 22):
            img[i, j] = 1.0
    for i in range(17, 20):
        for j in range(6, 22):
            img[i, j] = 1.0
    # Add some noise to make it more realistic
    img += np.random.randn(28, 28).astype(np.float32) * 0.05
    img = np.clip(img, 0.0, 1.0)
    return img


def try_load_mnist_idx():
    """Try to load a real MNIST test image from local files or download."""
    test_img_path = os.path.join(os.path.dirname(__file__), "t10k-images-idx3-ubyte.gz")
    if not os.path.exists(test_img_path):
        try:
            print("Downloading MNIST test set...", file=sys.stderr)
            urllib.request.urlretrieve(
                "https://github.com/cvdfoundation/mnist/raw/master/t10k-images-idx3-ubyte.gz",
                test_img_path
            )
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return None

    try:
        with gzip.open(test_img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read first image (digit '7' = label 7 at index 0)
            img_data = f.read(rows * cols)
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(rows, cols).astype(np.float32)
            img /= 255.0
            return img
    except Exception as e:
        print(f"MNIST parse failed: {e}", file=sys.stderr)
        return None


# ============================================================
# Encode ONNX protobuf directly (no onnx dependency)
# ============================================================
def encode_varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def encode_tag(fn: int, wire: int) -> bytes:
    return encode_varint((fn << 3) | wire)


def encode_varint_f(fn: int, v: int) -> bytes:
    return encode_tag(fn, 0) + encode_varint(v)


def encode_ld(fn: int, data: bytes) -> bytes:
    return encode_tag(fn, 2) + encode_varint(len(data)) + data


def encode_string(fn: int, s: str) -> bytes:
    return encode_ld(fn, s.encode('utf-8'))


def encode_int32(fn: int, v: int) -> bytes:
    return encode_varint_f(fn, v)


def encode_int64(fn: int, v: int) -> bytes:
    return encode_varint_f(fn, v)


def encode_float(fn: int, v: float) -> bytes:
    return encode_tag(fn, 5) + struct.pack('<f', v)


def encode_packed_ints(fn: int, vals: list) -> bytes:
    data = b''.join(encode_varint(v) for v in vals)
    return encode_ld(fn, data)


def encode_packed_floats(fn: int, vals: list) -> bytes:
    data = b''.join(struct.pack('<f', v) for v in vals)
    return encode_ld(fn, data)


# ============================================================
# Field numbers
# ============================================================
F_DIM_VALUE = 1
F_TENSORSHAPE_DIM = 1
F_TENSORSHAPE = 2  # inside TypeProto.Tensor
F_TYPE_TENSOR = 1  # TypeProto.tensor_type
F_VALUEINFO_NAME = 1
F_VALUEINFO_TYPE = 5
F_NODEPROTO_INPUT = 1
F_NODEPROTO_OUTPUT = 2
F_NODEPROTO_OP = 4
F_NODEPROTO_ATTR = 5
F_ATTR_NAME = 1
F_ATTR_INTS = 8
F_ATTR_I = 3
F_ATTR_F = 2
F_TENSORPROTO_DIMS = 1
F_TENSORPROTO_DTYPE = 2
F_TENSORPROTO_FLOAT_DATA = 6
F_TENSORPROTO_RAW = 9
F_TENSORPROTO_NAME = 8
F_GRAPHPROTO_NODE = 1
F_GRAPHPROTO_INIT = 5
F_GRAPHPROTO_INPUT = 11
F_GRAPHPROTO_OUTPUT = 12
F_GRAPHPROTO_VALUE_INFO = 13
F_MODELPROTO_GRAPH = 7
F_MODELPROTO_OPSET = 8
F_MODELPROTO_IR = 1
F_OPSET_DOMAIN = 1
F_OPSET_VERSION = 2

ONNX_FLOAT = 1
ONNX_INT64 = 7


def make_dim(v):
    return encode_ld(F_TENSORSHAPE_DIM, encode_int64(F_DIM_VALUE, v))


def make_shape(dims):
    return b''.join(make_dim(d) for d in dims)


def make_type(shape_dims):
    shape = make_shape(shape_dims)
    return encode_ld(F_TYPE_TENSOR, encode_ld(F_TENSORSHAPE, shape))


def make_value_info(name, dims):
    r = encode_string(F_VALUEINFO_NAME, name)
    r += encode_ld(F_VALUEINFO_TYPE, make_type(dims))
    return r


def make_tensor_proto(name, dims, float_vals, use_raw=True):
    r = b''
    if name:
        r += encode_string(F_TENSORPROTO_NAME, name)
    r += encode_packed_ints(F_TENSORPROTO_DIMS, dims)
    r += encode_int32(F_TENSORPROTO_DTYPE, ONNX_FLOAT)
    if use_raw:
        raw = b''.join(struct.pack('<f', v) for v in float_vals)
        r += encode_ld(F_TENSORPROTO_RAW, raw)
    else:
        r += encode_packed_floats(F_TENSORPROTO_FLOAT_DATA, float_vals)
    return r


def make_tensor_proto_int64(name, dims, int_vals):
    r = b''
    if name:
        r += encode_string(F_TENSORPROTO_NAME, name)
    r += encode_packed_ints(F_TENSORPROTO_DIMS, dims)
    r += encode_int32(F_TENSORPROTO_DTYPE, ONNX_INT64)
    raw = b''.join(struct.pack('<q', v) for v in int_vals)
    r += encode_ld(F_TENSORPROTO_RAW, raw)
    return r


def make_attr_ints(name, vals):
    return encode_string(F_ATTR_NAME, name) + encode_packed_ints(F_ATTR_INTS, vals)


def make_attr_int(name, v):
    return encode_string(F_ATTR_NAME, name) + encode_int64(F_ATTR_I, v)


def make_node(op_type, inputs, outputs, attrs=None):
    r = b''
    for inp in inputs:
        r += encode_string(F_NODEPROTO_INPUT, inp)
    for out in outputs:
        r += encode_string(F_NODEPROTO_OUTPUT, out)
    r += encode_string(F_NODEPROTO_OP, op_type)
    if attrs:
        for attr in attrs:
            r += encode_ld(F_NODEPROTO_ATTR, attr)
    return r


def make_graph(nodes, inputs, outputs, initializers=None, value_infos=None):
    r = b''
    for node in nodes:
        r += encode_ld(F_GRAPHPROTO_NODE, node)
    if initializers:
        for init in initializers:
            r += encode_ld(F_GRAPHPROTO_INIT, init)
    for inp in inputs:
        r += encode_ld(F_GRAPHPROTO_INPUT, inp)
    for out in outputs:
        r += encode_ld(F_GRAPHPROTO_OUTPUT, out)
    if value_infos:
        for vi in value_infos:
            r += encode_ld(F_GRAPHPROTO_VALUE_INFO, vi)
    return r


def make_opset_import(version=11):
    r = encode_string(F_OPSET_DOMAIN, "")
    r += encode_int64(F_OPSET_VERSION, version)
    return r


def make_model(graph_bytes, ir_version=6, opset_version=11):
    r = encode_int64(F_MODELPROTO_IR, ir_version)
    r += encode_ld(F_MODELPROTO_OPSET, make_opset_import(opset_version))
    r += encode_ld(F_MODELPROTO_GRAPH, graph_bytes)
    return r


# ============================================================
# Create the classification model
# ============================================================
def create_mnist_model():
    """Create an all-conv MNIST classifier using only supported ops.

    Architecture:
      Input(1,1,28,28) → Conv1(k=3,p=1, 1→4) → Relu → MaxPool(2×2)
      → Conv2(k=3,p=1, 4→8) → Relu → MaxPool(2×2)
      → GlobalAveragePool → Reshape(1,8) → MatMul(8→10) → Softmax
    """
    # Weights: use deterministic values for reproducibility
    rng = np.random.RandomState(42)

    # Conv1 weight: (4, 1, 3, 3)
    w1 = rng.randn(4, 1, 3, 3).astype(np.float32) * 0.3
    w1_tensor = make_tensor_proto("conv1.weight", [4, 1, 3, 3], w1.flatten().tolist())

    # Conv2 weight: (8, 4, 3, 3)
    w2 = rng.randn(8, 4, 3, 3).astype(np.float32) * 0.3
    w2_tensor = make_tensor_proto("conv2.weight", [8, 4, 3, 3], w2.flatten().tolist())

    # MatMul weight: (8, 10)
    w3 = rng.randn(8, 10).astype(np.float32) * 0.3
    w3_tensor = make_tensor_proto("fc.weight", [8, 10], w3.flatten().tolist())

    # Reshape target shape: [1, 8]
    reshape_shape = make_tensor_proto_int64("reshape_shape", [2], [1, 8])

    # Value info for intermediates
    input_vi = make_value_info("input", [1, 1, 28, 28])
    output_vi = make_value_info("output", [1, 10])

    # Nodes
    nodes = []

    # Conv1: (1,1,28,28) → (1,4,28,28) with pad=1, kernel=3
    nodes.append(make_node("Conv",
        ["input", "conv1.weight"],
        ["conv1_out"],
        [make_attr_ints("kernel_shape", [3, 3]),
         make_attr_ints("strides", [1, 1]),
         make_attr_ints("pads", [1, 1, 1, 1])]))

    nodes.append(make_node("Relu", ["conv1_out"], ["relu1_out"]))

    nodes.append(make_node("MaxPool",
        ["relu1_out"], ["pool1_out"],
        [make_attr_ints("kernel_shape", [2, 2]),
         make_attr_ints("strides", [2, 2]),
         make_attr_ints("pads", [0, 0, 0, 0])]))

    # Conv2: (1,4,14,14) → (1,8,14,14) with pad=1, kernel=3
    nodes.append(make_node("Conv",
        ["pool1_out", "conv2.weight"],
        ["conv2_out"],
        [make_attr_ints("kernel_shape", [3, 3]),
         make_attr_ints("strides", [1, 1]),
         make_attr_ints("pads", [1, 1, 1, 1])]))

    nodes.append(make_node("Relu", ["conv2_out"], ["relu2_out"]))

    # MaxPool2: (1,8,14,14) → (1,8,7,7)
    nodes.append(make_node("MaxPool",
        ["relu2_out"], ["pool2_out"],
        [make_attr_ints("kernel_shape", [2, 2]),
         make_attr_ints("strides", [2, 2]),
         make_attr_ints("pads", [0, 0, 0, 0])]))

    # GlobalAveragePool: (1,8,7,7) → (1,8,1,1)
    nodes.append(make_node("GlobalAveragePool", ["pool2_out"], ["gap_out"]))

    # Reshape: (1,8,1,1) → (1,8)
    nodes.append(make_node("Reshape",
        ["gap_out", "reshape_shape"], ["reshape_out"]))

    # MatMul: (1,8) × (8,10) → (1,10)
    nodes.append(make_node("MatMul",
        ["reshape_out", "fc.weight"], ["logits"]))

    # Softmax: (1,10) → (1,10)
    nodes.append(make_node("Softmax", ["logits"], ["output"],
        [make_attr_int("axis", 1)]))

    # Intermediates need value info
    value_infos = [
        make_value_info("conv1_out", [1, 4, 28, 28]),
        make_value_info("relu1_out", [1, 4, 28, 28]),
        make_value_info("pool1_out", [1, 4, 14, 14]),
        make_value_info("conv2_out", [1, 8, 14, 14]),
        make_value_info("relu2_out", [1, 8, 14, 14]),
        make_value_info("pool2_out", [1, 8, 7, 7]),
        make_value_info("gap_out", [1, 8, 1, 1]),
        make_value_info("reshape_out", [1, 8]),
        make_value_info("logits", [1, 10]),
    ]

    initializers = [w1_tensor, w2_tensor, w3_tensor, reshape_shape]

    graph = make_graph(nodes, [input_vi], [output_vi], initializers, value_infos)
    model = make_model(graph)
    return model, w1, w2, w3


# ============================================================
# Compute reference output using numpy
# ============================================================
def conv2d_ref(x, w, pad):
    N, C, H, W = x.shape
    K, _, KH, KW = w.shape
    OH = H + 2 * pad - KH + 1
    OW = W + 2 * pad - KW + 1
    out = np.zeros((N, K, OH, OW), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            for oh in range(OH):
                for ow in range(OW):
                    s = 0.0
                    for c in range(C):
                        for kh in range(KH):
                            for kw in range(KW):
                                ih = oh - pad + kh
                                iw = ow - pad + kw
                                if 0 <= ih < H and 0 <= iw < W:
                                    s += x[n, c, ih, iw] * w[k, c, kh, kw]
                    out[n, k, oh, ow] = s
    return out


def maxpool_ref(x, ks, stride):
    N, C, H, W = x.shape
    OH = (H - ks) // stride + 1
    OW = (W - ks) // stride + 1
    out = np.zeros((N, C, OH, OW), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    patch = x[n, c, oh*stride:oh*stride+ks, ow*stride:ow*stride+ks]
                    out[n, c, oh, ow] = np.max(patch)
    return out


def relu_ref(x):
    return np.maximum(x, 0)


def globalavgpool_ref(x):
    return x.mean(axis=(2, 3), keepdims=True)


def softmax_ref(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def compute_reference(img, w1, w2, w3):
    """Compute reference output of the model for comparison."""
    x = img.reshape(1, 1, 28, 28).astype(np.float32)

    # Conv1 + Relu + MaxPool
    x = conv2d_ref(x, w1, pad=1)
    x = relu_ref(x)
    x = maxpool_ref(x, ks=2, stride=2)  # (1,4,14,14)

    # Conv2 + Relu + MaxPool
    x = conv2d_ref(x, w2, pad=1)  # (1,8,14,14)
    x = relu_ref(x)
    x = maxpool_ref(x, ks=2, stride=2)  # (1,8,7,7)

    # GlobalAveragePool → (1,8,1,1)
    x = globalavgpool_ref(x)

    # Reshape → (1,8)
    x = x.reshape(1, 8)

    # MatMul → (1,10)
    x = x @ w3

    # Softmax
    x = softmax_ref(x)

    return x.flatten()


# ============================================================
# Main
# ============================================================
def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Get test image
    img = try_load_mnist_idx()
    if img is None:
        print("Using generated digit image", file=sys.stderr)
        img = generate_digit_image()
    else:
        print(f"Loaded real MNIST digit image ({img.shape[0]}x{img.shape[1]})", file=sys.stderr)

    print(f"Image stats: min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}", file=sys.stderr)

    # Create model
    model_data, w1, w2, w3 = create_mnist_model()

    # Save ONNX model
    onnx_path = os.path.join(output_dir, "mnist_classifier.onnx")
    with open(onnx_path, "wb") as f:
        f.write(model_data)
    print(f"Written {onnx_path} ({len(model_data)} bytes)", file=sys.stderr)

    # Compute reference output
    ref = compute_reference(img, w1, w2, w3)
    print(f"\nReference output (10 classes):", file=sys.stderr)
    for i, v in enumerate(ref):
        print(f"  Class {i}: {v:.6f}", file=sys.stderr)
    print(f"  Predicted class: {np.argmax(ref)} (prob={ref.max():.4f})", file=sys.stderr)

    # Save input as raw binary floats
    input_path = os.path.join(output_dir, "mnist_input.bin")
    input_data = img.astype(np.float32).reshape(1, 1, 28, 28).tobytes()
    with open(input_path, "wb") as f:
        f.write(input_data)
    print(f"Written {input_path} ({len(input_data)} bytes)", file=sys.stderr)

    # Save reference output
    ref_path = os.path.join(output_dir, "mnist_ref_output.bin")
    ref_data = ref.astype(np.float32).tobytes()
    with open(ref_path, "wb") as f:
        f.write(ref_data)
    print(f"Written {ref_path} ({len(ref_data)} bytes)", file=sys.stderr)

    # Print C embeddable bytes for the model
    print(f"\nstatic const unsigned char onnx_mnist_model[] = {{")
    for i in range(0, len(model_data), 16):
        chunk = model_data[i:i+16]
        hex_bytes = ', '.join(f'0x{b:02X}' for b in chunk)
        comma = ',' if i + 16 < len(model_data) else ''
        print(f"    {hex_bytes}{comma}")
    print(f"}};")
    print(f"static const size_t onnx_mnist_model_len = {len(model_data)};")


if __name__ == '__main__':
    main()
