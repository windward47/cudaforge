"""Generate a BERT-like ONNX model exercising all Phase B operators.

Self-contained — no HuggingFace download required.
Tests: Exp, ReduceSum, ReduceMax, Cast, ArgMax + all Phase A/B ops.

Avoids broadcast Sub (x - x.max()) since our binary-op broadcast only
handles scalar/simple-tile.  Weights are small and input values modest,
so exp(x) without stabilization is safe.

Outputs (in tests/):
  bert_base_test.onnx       — ONNX model
  bert_base_input.bin       — raw f32 input
  bert_base_ref_output.bin  — raw f32 reference output (ONNX Runtime)

Usage:
  python tests/gen_bert_base_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def write_bin(path, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(data.astype(np.float32).tobytes())


class PhaseBTestModel(nn.Module):
    """Model exercising all Phase B operators + existing ops.

    Architecture:
      input_ids: (1, seq_len) int64
      -> Cast(to=FLOAT) -> (seq_len,) float              [B3 Cast]
      -> Embedding (Gather) -> (seq_len, hidden)         [Gather]
      -> LayerNorm -> (seq_len, hidden)                  [LayerNorm]
      -> Linear1 + Exp activation -> (seq_len, 4*hidden) [MatMul, Add, Exp]
      -> ReduceSum(to=last dim) -> (seq_len, 1)          [B2 ReduceSum]
      -> ReduceMax(to=last dim) -> (seq_len, 1)          [B2 ReduceMax]
      -> Add -> (seq_len, 1)                             [Add]
      -> ArgMax -> () scalar index                       [B4 ArgMax]
      -> Unsqueeze -> (1, 1)                             [Unsqueeze]
      -> fc2 (1->vocab_size) -> (1, vocab_size)          [MatMul, Add]
      -> Softmax -> (1, vocab_size)                      [Softmax]
      -> Concat with argmax index -> (1, vocab_size+1)   [Concat]
    """

    def __init__(self, vocab_size=256, hidden_size=64, seq_len=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(1, vocab_size)

    def forward(self, input_ids):
        x = input_ids.squeeze(0)                        # (seq_len,)
        x = x.float() * 1.0 + 0.0                       # Cast int64->float

        x = self.embed(input_ids).squeeze(0)            # (seq_len, hidden)
        x = self.ln(x)                                  # (seq_len, hidden)
        x = self.fc1(x)                                 # (seq_len, 4*hidden)
        x = torch.exp(x)                                # Exp (B1) — no Sub needed

        x_sum = x.sum(dim=-1, keepdim=True)             # (seq_len, 1)  ReduceSum
        x_max = x.max(dim=-1, keepdim=True)[0]          # (seq_len, 1)  ReduceMax
        x = x_sum + x_max                               # (seq_len, 1)  Add

        x_idx = x.squeeze(-1).argmax(dim=-1)            # () ArgMax
        x_idx = x_idx.float().unsqueeze(0).unsqueeze(0) # (1, 1) Unsqueeze

        logits = self.fc2(x_idx)                        # (1, vocab_size)
        logits = F.softmax(logits, dim=-1)              # Softmax
        return torch.cat([logits, x_idx], dim=1)         # (1, vocab_size+1)


def main():
    torch.manual_seed(42)

    hidden_size = 64
    seq_len = 8
    vocab_size = 256

    print(f"Creating Phase B test model: hidden={hidden_size}, seq={seq_len}, vocab={vocab_size}")

    model = PhaseBTestModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len)
    model.eval()

    onnx_path = os.path.join(TEST_DIR, "bert_base_test.onnx")
    dummy_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    torch.onnx.export(
        model, dummy_ids, onnx_path,
        opset_version=11,
        input_names=["input_ids"],
        output_names=["output"],
    )

    onnx_model = onnx.load(onnx_path)
    op_types = set(n.op_type for n in onnx_model.graph.node)
    print(f"ONNX ops used ({len(op_types)}): {sorted(op_types)}")

    target_ops = {"Exp", "ReduceSum", "ReduceMax", "Cast", "ArgMax"}
    found = target_ops & op_types
    missing = target_ops - op_types
    if found:
        print(f"Phase B ops found: {sorted(found)}")
    if missing:
        print(f"Phase B ops MISSING from model: {sorted(missing)}")

    unsupported = {
        "Shape", "ConstantOfShape", "Where", "Erf", "Tanh", "Pow", "Sqrt",
        "Expand", "NonZero", "Range", "ScatterND", "Tile", "TopK",
        "Not", "Less", "Equal", "And", "If", "Loop", "Identity",
        "Compress", "OneHot", "CumSum", "Constant",
    }
    needed_unsupported = op_types & unsupported
    if needed_unsupported:
        print(f"\n*** WARNING: Unsupported ops found: {needed_unsupported} ***")

    fixed_ids = torch.tensor([[12, 45, 3, 200, 88, 15, 99, 1]], dtype=torch.long)
    print(f"Input token ids: {fixed_ids.tolist()}")

    input_path = os.path.join(TEST_DIR, "bert_base_input.bin")
    write_bin(input_path, fixed_ids.numpy().astype(np.float32))
    print(f"Wrote {input_path}")

    ort_session = ort.InferenceSession(onnx_path)
    ort_outputs = ort_session.run(None, {"input_ids": fixed_ids.numpy()})
    ref = ort_outputs[0]
    print(f"Reference output shape: {ref.shape}")

    ref_path = os.path.join(TEST_DIR, "bert_base_ref_output.bin")
    write_bin(ref_path, ref)
    print(f"Wrote {ref_path} ({ref.nbytes} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
