"""Create a simple ONNX model using only ops supported by our engine:
  Conv, ReLU, MaxPool, Gemm, Sigmoid, BatchNorm, AveragePool, Gelu

Creates: test_pipeline.onnx — a multi-op pipeline with known weights.
Also computes reference outputs using numpy.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

# ============================================================
# Model: mini CNN pipeline with only supported ops
#
#   Input(1,1,4,4)
#     → Conv(kernel=2, K=2, pad=0)  → (1,2,3,3)
#     → ReLU                        → (1,2,3,3)
#     → MaxPool(kernel=2, stride=2) → (1,2,1,1)
#     → Flatten                     → (1,2)   -- unsupported, skip
#
# Simplified: Input(1,1,4,4) → Conv → ReLU → Output(1,2,3,3)
# ============================================================

def make_conv_relu_model():
    """Input(1,1,4,4) → Conv(kernel=2, K=2) → ReLU → Output(1,2,3,3)"""
    # Weight: shape (2, 1, 2, 2), values = [1,2, 3,4,  5,6, 7,8]
    weight_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32).reshape(2, 1, 2, 2)

    weight = helper.make_tensor(
        name="conv1.weight",
        data_type=TensorProto.FLOAT,
        dims=[2, 1, 2, 2],
        vals=weight_vals.flatten().tolist()
    )

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 3])

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv1.weight"],
        outputs=["conv_out"],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_out"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[conv_node, relu_node],
        name="test_pipeline",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=[weight],
    )

    model = helper.make_model(graph, producer_name="test_gen")
    model.opset_import[0].version = 11
    return model


def make_relu_model():
    """Input(1,4) → ReLU → Output(1,4)"""
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    node = helper.make_node("Relu", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test_relu", [input_vi], [output_vi])
    model = helper.make_model(graph, producer_name="test_gen")
    model.opset_import[0].version = 11
    return model


def compute_conv_ref(input_data, weight_data):
    """Manual conv2d: input(1,1,H,W) @ weight(K,1,KH,KW) with stride=1, pad=0"""
    H, W = input_data.shape[2], input_data.shape[3]
    K, _, KH, KW = weight_data.shape
    OH = H - KH + 1
    OW = W - KW + 1
    out = np.zeros((1, K, OH, OW), dtype=np.float32)
    for k in range(K):
        for i in range(OH):
            for j in range(OW):
                patch = input_data[0, 0, i:i+KH, j:j+KW]
                out[0, k, i, j] = np.sum(patch * weight_data[k, 0])
    return out


def main():
    model = make_conv_relu_model()
    path = "tests/test_pipeline.onnx"
    onnx.save(model, path)
    print(f"Written {path}")

    # Compute reference output
    input_data = np.array([[[[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]]]], dtype=np.float32)

    weight_data = np.array([[[[1, 2],
                               [3, 4]]],
                             [[[5, 6],
                               [7, 8]]]], dtype=np.float32)

    conv_out = compute_conv_ref(input_data, weight_data)
    relu_out = np.maximum(conv_out, 0)  # all positive anyway

    print(f"Input (1,1,4,4):\n{input_data[0,0]}")
    print(f"Weight K=0:\n{weight_data[0,0]}")
    print(f"Weight K=1:\n{weight_data[1,0]}")
    print(f"Conv output (1,2,3,3), K=0:\n{conv_out[0,0]}")
    print(f"Conv output (1,2,3,3), K=1:\n{conv_out[0,1]}")
    print(f"ReLU output (same, all positive):\n{relu_out}")
    print(f"Expected output flat: {relu_out.flatten().tolist()}")

    # Print C embeddable bytes
    with open(path, "rb") as f:
        data = f.read()
    print(f"\nstatic const unsigned char onnx_test_pipeline_data[] = {{")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_bytes = ', '.join(f'0x{b:02X}' for b in chunk)
        if i + 16 < len(data):
            print(f"    {hex_bytes},")
        else:
            print(f"    {hex_bytes}")
    print(f"}};")
    print(f"static const size_t onnx_test_pipeline_data_len = {len(data)};")


if __name__ == "__main__":
    main()
