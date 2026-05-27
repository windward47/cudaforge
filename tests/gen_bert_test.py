"""Generate a BERT-like ONNX model + test data for end-to-end verification.

Exercises LayerNorm, Gather (Embedding), Squeeze/Unsqueeze, and existing ops.
Uses 2D tensors throughout to avoid N-D batched MatMul (not yet supported).

Usage:
  python tests/gen_bert_test.py

Outputs (in tests/):
  bert_test.onnx        — ONNX model
  bert_input.bin        — raw f32 input token IDs (int64→f32)
  bert_ref_output.bin   — raw f32 reference output (onnxruntime)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class BertTestFFN(nn.Module):
    """2-layer FFN with residual connections + LayerNorm + SiLU.

    Operates entirely on 2D tensors: (seq_len, hidden_size).
    Exercises: LayerNorm(2×), MatMul+Add(linear), SiLU(Sigmoid+Mul), Add(residual)
    """

    def __init__(self, hidden_size=64, intermediate_size=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(intermediate_size, eps=1e-5)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x):
        # Block 1: LN → Linear → SiLU
        identity = x
        x = self.ln1(x)
        x = F.silu(self.fc1(x))          # SiLU = Sigmoid + Mul
        # Block 2: LN → Linear → residual
        x = self.ln2(x)
        x = F.silu(self.fc2(x))
        x = x + identity                 # Residual add
        x = self.ln3(x)
        return x


class BertTestModel(nn.Module):
    """Minimal BERT-like model using only 2D MatMul.

    Architecture: Embedding → Squeeze(batch) → FFN → LM head

    Exercises:
      - nn.Embedding → Gather (ONNX op)
      - LayerNorm (3×)
      - Linear → MatMul(2D) + Add
      - SiLU activation (Sigmoid + Mul)
      - Add (residual)
      - Unsqueeze / Squeeze (round-trip on 2D tensor)
    """

    def __init__(self, vocab_size=256, hidden_size=64, seq_len=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.ffn = BertTestFFN(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: (1, seq_len) INT64
        x = self.embed(input_ids)          # (1, seq, hidden) — Gather
        x = x.squeeze(0)                   # (seq, hidden) — 2D, avoids N-D matmul

        # Squeeze/Unsqueeze round-trip on 2D tensor
        x = x.unsqueeze(1)                 # (seq, 1, hidden)
        x = x.squeeze(1)                   # (seq, hidden)

        # FFN with residual + LayerNorm
        x = self.ffn(x)                    # (seq, hidden)

        # LM head → logits
        logits = self.lm_head(x)           # (seq, vocab_size)
        return logits


def write_bin(path, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(data.astype(np.float32).tobytes())


def main():
    torch.manual_seed(42)

    hidden_size = 64
    seq_len = 8
    vocab_size = 256

    print(f"Creating 2D BERT-like test model: hidden={hidden_size}, seq={seq_len}, vocab={vocab_size}")

    model = BertTestModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len)
    model.eval()

    onnx_path = os.path.join(TEST_DIR, "bert_test.onnx")
    dummy_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    torch.onnx.export(
        model, dummy_ids, onnx_path,
        opset_version=18,
        input_names=["input_ids"],
        output_names=["logits"],
    )

    onnx_model = onnx.load(onnx_path)
    op_types = set(n.op_type for n in onnx_model.graph.node)
    print(f"ONNX ops used: {sorted(op_types)}")

    fixed_ids = torch.tensor([[12, 45, 3, 200, 88, 15, 99, 1]], dtype=torch.long)
    print(f"Input token ids: {fixed_ids.tolist()}")

    ort_session = ort.InferenceSession(onnx_path)
    ort_outputs = ort_session.run(None, {"input_ids": fixed_ids.numpy()})
    ref = ort_outputs[0]
    print(f"Reference output shape: {ref.shape}")

    input_path = os.path.join(TEST_DIR, "bert_input.bin")
    ref_path = os.path.join(TEST_DIR, "bert_ref_output.bin")
    write_bin(input_path, fixed_ids.numpy().astype(np.float32))
    write_bin(ref_path, ref)
    print(f"Wrote {input_path} ({fixed_ids.numpy().nbytes} B)")
    print(f"Wrote {ref_path} ({ref.nbytes} B)")
    print("Done.")


if __name__ == "__main__":
    main()
