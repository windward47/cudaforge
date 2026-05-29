"""Generate a small GPT-2-like decoder model for testing.

Small dims: hidden=64, heads=4, head_dim=16, seq_len=8, 2 layers.
Causal attention with torch.tril mask.
Exports ONNX, generates reference output via ONNX Runtime.

Outputs:
  tests/gpt2_test.onnx       - ONNX model
  tests/gpt2_input.bin        - raw f32 input (B, S, hidden)
  tests/gpt2_ref_output.bin   - raw f32 reference output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

BATCH = 1
SEQ_LEN = 8
HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
FF_DIM = HIDDEN * 4  # 256


def write_bin(path, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(data.astype(np.float32).tobytes())


class GPT2Block(nn.Module):
    """Single GPT-2 decoder block: LN → CausalMHA → Residual → LN → FFN(GELU) → Residual"""

    def __init__(self, hidden, heads, head_dim, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff1 = nn.Linear(hidden, ff_dim)
        self.ff2 = nn.Linear(ff_dim, hidden)

    def forward(self, x):
        B, S, D = x.shape
        H = NUM_HEADS
        d = HEAD_DIM

        # Pre-norm
        h = self.ln1(x)

        # QKV projections
        q = self.q_proj(h).reshape(B, S, H, d).transpose(1, 2)  # (B, H, S, d)
        k = self.k_proj(h).reshape(B, S, H, d).transpose(1, 2)
        v = self.v_proj(h).reshape(B, S, H, d).transpose(1, 2)

        # Causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)

        # Merge heads and output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = x + attn_out

        # FFN with residual
        h2 = self.ln2(x)
        h2 = F.gelu(self.ff1(h2))
        h2 = self.ff2(h2)
        x = x + h2

        return x


class GPT2Small(nn.Module):
    """2-layer GPT-2 decoder."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            GPT2Block(HIDDEN, NUM_HEADS, HEAD_DIM, FF_DIM) for _ in range(2)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Creating GPT-2 small model:")
    print(f"  hidden={HIDDEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}, ff_dim={FF_DIM}")
    print(f"  batch={BATCH}, seq_len={SEQ_LEN}, layers=2")

    model = GPT2Small()
    model.eval()

    onnx_path = os.path.join(TEST_DIR, "gpt2_test.onnx")
    dummy = torch.randn(BATCH, SEQ_LEN, HIDDEN)

    # Export ONNX
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed dims
    )

    # Check exported model
    onnx_model = onnx.load(onnx_path)
    op_types = set(n.op_type for n in onnx_model.graph.node)
    print(f"ONNX ops used ({len(op_types)}): {sorted(op_types)}")

    # Generate input and reference
    fixed_input = torch.tensor(
        np.random.RandomState(42).randn(BATCH, SEQ_LEN, HIDDEN).astype(np.float32)
    )
    print(f"Input shape: {fixed_input.shape}, range: [{fixed_input.min():.3f}, {fixed_input.max():.3f}]")

    input_path = os.path.join(TEST_DIR, "gpt2_input.bin")
    write_bin(input_path, fixed_input.numpy())
    print(f"Wrote {input_path}")

    # ONNX Runtime reference
    ort_session = ort.InferenceSession(onnx_path)
    ort_outputs = ort_session.run(None, {"input": fixed_input.numpy()})
    ref = ort_outputs[0]
    print(f"Reference output shape: {ref.shape}, range: [{ref.min():.6f}, {ref.max():.6f}]")

    ref_path = os.path.join(TEST_DIR, "gpt2_ref_output.bin")
    write_bin(ref_path, ref)
    print(f"Wrote {ref_path} ({ref.nbytes} bytes)")

    # Verify PyTorch vs ONNX Runtime
    with torch.no_grad():
        pt_out = model(fixed_input).numpy()
    max_diff = np.abs(pt_out - ref).max()
    print(f"PyTorch vs ONNX Runtime max_diff: {max_diff:.2e}")
    if max_diff > 1e-4:
        print("  WARNING: PyTorch/ORT mismatch — possible export issue")

    # Print graph structure
    print("\nONNX graph nodes:")
    for i, node in enumerate(onnx_model.graph.node):
        inputs_str = ", ".join(node.input[:3])
        if len(node.input) > 3:
            inputs_str += ", ..."
        outputs_str = ", ".join(node.output[:2])
        if len(node.output) > 2:
            outputs_str += ", ..."
        print(f"  [{i:2d}] {node.op_type:12s}  {inputs_str:60s} → {outputs_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
