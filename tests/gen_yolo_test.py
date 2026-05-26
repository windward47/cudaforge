"""Export YOLOv8n ONNX model and generate test data using real weights."""
import torch
import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load YOLOv8n with real pretrained weights ---
print("Loading YOLOv8n (real COCO weights)...")
from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

# Export ONNX via ultralytics (uses TorchScript path, compatible with PyTorch 2.x)
onnx_path = os.path.join(OUT_DIR, "yolov8n_test.onnx")
print(f"Exporting ONNX via ultralytics to {onnx_path}...")
yolo.export(format="onnx", imgsz=640, opset=11, simplify=False)
# ultralytics exports to 'yolov8n.onnx' in current dir
default_onnx = os.path.join(os.getcwd(), "yolov8n.onnx")
if os.path.exists(default_onnx) and default_onnx != onnx_path:
    import shutil
    shutil.move(default_onnx, onnx_path)
    print(f"Moved {default_onnx} -> {onnx_path}")

print(f"ONNX model saved ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

# --- 2. Create test input ---
torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 640, 640)
input_np = dummy_input.numpy().astype(np.float32)
print(f"Input shape: {input_np.shape}")

# --- 3. Dump model ops ---
print("\n--- ONNX Model Ops ---")
import onnx
onnx_model = onnx.load(onnx_path)
op_types = set()
for node in onnx_model.graph.node:
    op_types.add(node.op_type)
print(f"Unique op types ({len(op_types)}): {sorted(op_types)}")
print(f"Total nodes: {len(onnx_model.graph.node)}")

graph = onnx_model.graph
for inp in graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  Input:  {inp.name} shape={shape}")
for out in graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name} shape={shape}")

supported = {
    "Conv", "Relu", "Sigmoid", "SiLU", "BatchNormalization",
    "MaxPool", "AveragePool", "GlobalAveragePool", "Add", "Mul",
    "Reshape", "Concat", "Resize", "Transpose", "Gemm", "MatMul",
    "Softmax", "Flatten", "Split", "Slice", "Unsqueeze", "Constant",
    "Sub", "Div", "Pow", "Sqrt", "ReduceMax", "Exp", "Cast", "Shape",
    "Gather", "Tile", "Range", "ScatterND",
}
unsupported = op_types - supported
if unsupported:
    print(f"\n*** UNSUPPORTED OPS: {unsupported} ***")
    for node in onnx_model.graph.node:
        if node.op_type in unsupported:
            print(f"  Node '{node.name}': op_type={node.op_type}")
else:
    print("All ops appear supported.")

# --- 4. Generate reference output via ONNX Runtime ---
print("\n--- Generating reference via ONNX Runtime ---")
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path)
ort_inputs = {sess.get_inputs()[0].name: input_np}
ort_outputs = sess.run(None, ort_inputs)
ref_np = ort_outputs[0].astype(np.float32)

print(f"Reference output shape: {ref_np.shape}")
print(f"Reference output range: [{ref_np.min():.6f}, {ref_np.max():.6f}]")
print(f"Reference output mean: {ref_np.mean():.6f}, std: {ref_np.std():.6f}")

# --- 5. Save input and reference output ---
input_path = os.path.join(OUT_DIR, "yolov8n_input.bin")
ref_path = os.path.join(OUT_DIR, "yolov8n_ref_output.bin")

input_np.tofile(input_path)
ref_np.tofile(ref_path)

print(f"\nInput saved: {input_path} ({input_np.nbytes} bytes, shape={input_np.shape})")
print(f"Reference saved: {ref_path} ({ref_np.nbytes} bytes, shape={ref_np.shape})")

# --- 6. Quick PyTorch verification (use predict API instead of raw model) ---
print("\n--- PyTorch verification (via predict API) ---")
results = yolo.predict(dummy_input, imgsz=640, verbose=False)
if len(results) > 0 and results[0].boxes is not None:
    n_det = len(results[0].boxes)
    print(f"Detections: {n_det} objects found")
    if n_det > 0:
        print(f"  Sample detection: {results[0].boxes.xyxy[0].tolist()}")
        print(f"  Sample confidence: {results[0].boxes.conf[0].item():.4f}")
else:
    print("No detections (expected for random noise input)")

print("\nDone. Files:")
for f in [onnx_path, input_path, ref_path]:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")
