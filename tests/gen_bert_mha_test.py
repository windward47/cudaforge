"""Generate a single BERT encoder layer ONNX model with real Multi-Head Attention.

Self-contained — no HuggingFace required.
Uses fixed-dimension reshape (no Shape/Constant ops) for compatibility.

Exercises: MatMul+Add(QKV) → Reshape×3 → Transpose×3 → MatMul(Q·K^T) → Mul(1/√d)
→ Softmax → MatMul(×V) → Transpose → Reshape → MatMul+Add(output_proj)
+ LayerNorm pre-norm + residual Add

Outputs (in tests/):
  bert_mha_test.onnx       — ONNX model
  bert_mha_input.bin       — raw f32 input
  bert_mha_ref_output.bin  — raw f32 reference output (ONNX Runtime)

Usage:
  python tests/gen_bert_mha_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
import os
from onnx import helper, numpy_helper

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed dimensions — must match the generated model
BATCH = 1
SEQ_LEN = 8
HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS  # 64


def write_bin(path, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(data.astype(np.float32).tobytes())


class BertSelfAttention(nn.Module):
    """Single BERT encoder self-attention layer with pre-LayerNorm.

    Input:  (1, 8, 768) f32 — fixed dims
    Output: (1, 8, 768) f32 (with residual connection)
    """

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HIDDEN, eps=1e-12)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        # x: (1, 8, 768) — fixed dims
        identity = x
        x = self.ln(x)

        # QKV projections
        q = self.q_proj(x)  # (1, 8, 768)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Multi-head split: (1, 8, 768) → (1, 12, 8, 64)
        # Use explicit fixed dims — avoids Shape op in ONNX export
        q = q.reshape(1, 8, 12, 64).transpose(1, 2)
        k = k.reshape(1, 8, 12, 64).transpose(1, 2)
        v = v.reshape(1, 8, 12, 64).transpose(1, 2)

        # Scaled dot-product attention
        scale = 64 ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (1,12,8,8)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)                       # (1,12,8,64)

        # Merge heads: (1, 12, 8, 64) → (1, 8, 768)
        attn_out = attn_out.transpose(1, 2).reshape(1, 8, 768)

        # Output projection + residual
        out = self.out_proj(attn_out) + identity
        return out


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Creating BERT self-attention model:")
    print(f"  hidden={HIDDEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
    print(f"  batch={BATCH}, seq_len={SEQ_LEN}")

    model = BertSelfAttention()
    model.eval()

    onnx_path = os.path.join(TEST_DIR, "bert_mha_test.onnx")
    dummy = torch.randn(BATCH, SEQ_LEN, HIDDEN)

    # Export with fixed dims only — avoids dynamic Shape ops
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    # Check exported ONNX ops
    onnx_model = onnx.load(onnx_path)

    # Convert to self-contained model (no external data) for C loader compat
    import tempfile
    tmp = os.path.join(TEST_DIR, "_bert_mha_tmp.onnx")
    onnx.save(onnx_model, tmp)
    os.replace(tmp, onnx_path)
    # Remove external data file if present
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Reload to verify
    onnx_model = onnx.load(onnx_path)
    op_types = set(n.op_type for n in onnx_model.graph.node)
    print(f"ONNX ops used ({len(op_types)}): {sorted(op_types)}")

    # Check for unsupported ops
    unsupported = {
        "Shape", "ConstantOfShape", "Where", "Erf", "Tanh", "Pow", "Sqrt",
        "Expand", "NonZero", "Range", "ScatterND", "Tile", "TopK",
        "Not", "Less", "Equal", "And", "If", "Loop", "Identity",
        "Compress", "OneHot", "CumSum", "Constant",
    }
    needed_unsupported = op_types & unsupported
    if needed_unsupported:
        print(f"\n*** WARNING: Unsupported ops found: {needed_unsupported} ***")

    # Generate input and reference
    fixed_input = torch.tensor(
        np.random.RandomState(42).randn(BATCH, SEQ_LEN, HIDDEN).astype(np.float32)
    )
    print(f"Input shape: {fixed_input.shape}, range: [{fixed_input.min():.3f}, {fixed_input.max():.3f}]")

    input_path = os.path.join(TEST_DIR, "bert_mha_input.bin")
    write_bin(input_path, fixed_input.numpy())
    print(f"Wrote {input_path}")

    # ONNX Runtime reference
    ort_session = ort.InferenceSession(onnx_path)
    ort_outputs = ort_session.run(None, {"input": fixed_input.numpy()})
    ref = ort_outputs[0]
    print(f"Reference output shape: {ref.shape}, range: [{ref.min():.6f}, {ref.max():.6f}]")

    ref_path = os.path.join(TEST_DIR, "bert_mha_ref_output.bin")
    write_bin(ref_path, ref)
    print(f"Wrote {ref_path} ({ref.nbytes} bytes)")

    # Quick sanity: verify PyTorch output matches ONNX Runtime
    with torch.no_grad():
        pt_out = model(fixed_input).numpy()
    max_diff = np.abs(pt_out - ref).max()
    print(f"PyTorch vs ONNX Runtime max_diff: {max_diff:.2e}")
    if max_diff > 1e-4:
        print("  WARNING: PyTorch/ORT mismatch — possible export issue")

    # Print model graph structure for reference
    print("\nONNX graph nodes:")
    for i, node in enumerate(onnx_model.graph.node):
        inputs_str = ", ".join(node.input[:3])
        if len(node.input) > 3:
            inputs_str += ", ..."
        outputs_str = ", ".join(node.output[:2])
        if len(node.output) > 2:
            outputs_str += ", ..."
        print(f"  [{i:2d}] {node.op_type:12s}  {inputs_str:60s} → {outputs_str}")

    print("Done.")


if __name__ == "__main__":
    main()
