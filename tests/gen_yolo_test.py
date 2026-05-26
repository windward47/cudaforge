"""Export YOLOv8n ONNX model and generate test data."""
import torch
import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load YOLOv8n ---
print("Loading YOLOv8n...")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    torch_model = model.model.model  # the underlying nn.Module
    torch_model.eval()
    print("Loaded via ultralytics")
except ImportError:
    print("ultralytics not available, loading torch hub...")
    torch_model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov8n', source='github')
    torch_model.eval()

# --- 2. Create test input ---
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 640, 640)
print(f"Input shape: {dummy_input.shape}")

# --- 3. Run PyTorch inference ---
print("Running PyTorch inference (this may take a moment)...")
with torch.no_grad():
    ref_output = model(dummy_input)
    if hasattr(ref_output, 'boxes'):
        # ultralytics YOLO returns a Results object; extract raw output
        ref_output = ref_output[0].boxes.data  # not raw; use the model directly
        # Actually, redo with torch_model directly
        ref_output = torch_model(dummy_input)
        if isinstance(ref_output, (list, tuple)):
            ref_output = ref_output[0]

if isinstance(ref_output, (list, tuple)):
    print(f"Model has {len(ref_output)} outputs")
    for i, o in enumerate(ref_output):
        print(f"  Output {i}: shape={o.shape}")
    # For YOLO detection, take the main detection output
    if len(ref_output) > 0:
        ref_output = ref_output[0]  # Use first output
else:
    print(f"Output shape: {ref_output.shape}")

if hasattr(ref_output, 'detach'):
    ref_output = ref_output.detach()

# --- 4. Export to ONNX ---
onnx_path = os.path.join(OUT_DIR, "yolov8n_test.onnx")
print(f"\nExporting ONNX model to {onnx_path}...")
torch.onnx.export(
    torch_model,
    dummy_input,
    onnx_path,
    input_names=["images"],
    output_names=["output0"],
    opset_version=11,
    dynamic_axes=None,
    export_params=True,
)
print(f"ONNX model saved ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

# --- 5. Dump model op types ---
print("\n--- ONNX Model Ops ---")
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    op_types = set()
    for node in onnx_model.graph.node:
        op_types.add(node.op_type)
    print(f"Unique op types: {sorted(op_types)}")
    print(f"Total nodes: {len(onnx_model.graph.node)}")
    supported = {
        "Conv", "Relu", "Sigmoid", "SiLU", "BatchNormalization",
        "MaxPool", "AveragePool", "GlobalAveragePool", "Add", "Mul",
        "Reshape", "Concat", "Resize", "Transpose", "Gemm", "MatMul",
        "Softmax", "Flatten"
    }
    unsupported = op_types - supported
    if unsupported:
        print(f"\n*** POTENTIALLY UNSUPPORTED OPS: {unsupported} ***")
        for node in onnx_model.graph.node:
            if node.op_type in unsupported:
                print(f"  Node '{node.name}': op_type={node.op_type}")
    else:
        print("All ops appear to be in the known set.")
except ImportError:
    print("(onnx not installed, skipping op check)")

# --- 6. Save input and reference output ---
ref_np = ref_output.numpy().astype(np.float32)
input_np = dummy_input.numpy().astype(np.float32)

input_path = os.path.join(OUT_DIR, "yolov8n_input.bin")
ref_path = os.path.join(OUT_DIR, "yolov8n_ref_output.bin")

input_np.tofile(input_path)
ref_np.tofile(ref_path)

print(f"\nInput saved to {input_path} ({input_np.nbytes} bytes, shape={input_np.shape})")
print(f"Reference output saved to {ref_path} ({ref_np.nbytes} bytes, shape={ref_np.shape})")

# --- 7. ONNX Runtime verification ---
try:
    import onnxruntime as ort
    print("\n--- ONNX Runtime verification ---")
    sess = ort.InferenceSession(onnx_path)
    ort_inputs = {sess.get_inputs()[0].name: input_np}
    ort_output = sess.run(None, ort_inputs)[0]
    diff = np.max(np.abs(ort_output - ref_np))
    print(f"Max difference vs PyTorch: {diff:.2e}")
    if diff < 1e-3:
        print("ONNX Runtime output matches PyTorch reference")
    else:
        print("WARNING: ONNX Runtime output differs from PyTorch!")
except ImportError:
    print("\n(onnxruntime not installed, skipping verification)")

print("\nDone. Files generated:")
for f in [onnx_path, input_path, ref_path]:
    if os.path.exists(f):
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
