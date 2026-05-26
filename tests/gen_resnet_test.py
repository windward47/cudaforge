"""Export ResNet-18 ONNX model and generate test data."""
import torch
import torchvision
import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load pretrained ResNet-18 ---
print("Loading pretrained ResNet-18...")
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# --- 2. Create a test input (random image, not a real photo) ---
# Use a fixed seed for reproducibility
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224)
print(f"Input shape: {dummy_input.shape}")

# --- 3. Run PyTorch inference to get reference output ---
print("Running PyTorch inference...")
with torch.no_grad():
    ref_output = model(dummy_input)
print(f"Output shape: {ref_output.shape}")

# Show top-5 predictions
probs = torch.softmax(ref_output[0], dim=0)
top5_prob, top5_idx = torch.topk(probs, 5)
print("\nTop-5 predictions (reference):")
for i in range(5):
    print(f"  Class {top5_idx[i].item():>5d}: prob={top5_prob[i].item():.6f}")

# --- 4. Export to ONNX ---
onnx_path = os.path.join(OUT_DIR, "resnet18_test.onnx")
print(f"\nExporting ONNX model to {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,  # ResNet uses Flatten which is supported in opset 11+
    dynamic_axes=None,
)
print(f"ONNX model saved ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

# --- 5. Save input and reference output ---
input_path = os.path.join(OUT_DIR, "resnet18_input.bin")
ref_path = os.path.join(OUT_DIR, "resnet18_ref_output.bin")

dummy_input_np = dummy_input.numpy().astype(np.float32)
ref_output_np = ref_output.numpy().astype(np.float32)

dummy_input_np.tofile(input_path)
ref_output_np.tofile(ref_path)
print(f"\nInput saved to {input_path} ({dummy_input_np.nbytes} bytes)")
print(f"Reference output saved to {ref_path} ({ref_output_np.nbytes} bytes)")

# --- 6. Also run ONNX Runtime verification if available ---
try:
    import onnxruntime as ort
    print("\n--- ONNX Runtime verification ---")
    sess = ort.InferenceSession(onnx_path)
    ort_inputs = {sess.get_inputs()[0].name: dummy_input_np}
    ort_output = sess.run(None, ort_inputs)[0]
    diff = np.max(np.abs(ort_output - ref_output_np))
    print(f"Max difference vs PyTorch: {diff:.2e}")
    if diff < 1e-3:
        print("ONNX Runtime output matches PyTorch reference ✓")
    else:
        print("WARNING: ONNX Runtime output differs from PyTorch!")
except ImportError:
    print("\n(onnxruntime not installed, skipping verification)")

# --- 7. Dump model op types ---
print("\n--- ONNX Model Ops ---")
import onnx
onnx_model = onnx.load(onnx_path)
op_types = set()
for node in onnx_model.graph.node:
    op_types.add(node.op_type)
print(f"Unique op types: {sorted(op_types)}")
print(f"Total nodes: {len(onnx_model.graph.node)}")

# Check for unsupported ops
supported = {
    "Conv", "Relu", "BatchNormalization", "MaxPool", "AveragePool",
    "GlobalAveragePool", "Add", "Reshape", "Gemm", "MatMul", "Softmax", "Flatten"
}
unsupported = op_types - supported
if unsupported:
    print(f"\n*** UNSUPPORTED OPS: {unsupported} ***")
    for node in onnx_model.graph.node:
        if node.op_type in unsupported:
            print(f"  Node '{node.name}': op_type={node.op_type}")
else:
    print("All ops are in the known set.")
