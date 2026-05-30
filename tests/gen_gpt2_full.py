"""Generate a full GPT-2-like model for end-to-end LLM inference testing.

Includes: token embedding + position embedding + 2 transformer layers
+ final LayerNorm + LM head (logits projection).

Small dims: vocab=256, hidden=64, heads=4, head_dim=16, ff_dim=256, layers=2, max_seq=32

Outputs:
  tests/gpt2_full.onnx        - ONNX model
  tests/gpt2_full_vocab.txt   - simple vocabulary file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Small model dimensions
VOCAB = 256
HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
FF_DIM = HIDDEN * 4  # 256
LAYERS = 2
MAX_SEQ = 32


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block: LN → CausalMHA → Residual → LN → FFN → Residual"""

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
        H = HEADS
        d = HEAD_DIM

        # Pre-norm attention
        h = self.ln1(x)
        q = self.q_proj(h).reshape(B, S, H, d).transpose(1, 2)
        k = self.k_proj(h).reshape(B, S, H, d).transpose(1, 2)
        v = self.v_proj(h).reshape(B, S, H, d).transpose(1, 2)

        # Causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.out_proj(attn_out)

        # Pre-norm FFN (use tanh-based GELU to avoid Erf/Where ops)
        h2 = self.ln2(x)
        h2 = self.ff1(h2)
        h2 = 0.5 * h2 * (1.0 + torch.tanh(0.7978845608028654 * (h2 + 0.044715 * h2 * h2 * h2)))
        h2 = self.ff2(h2)
        x = x + h2

        return x


class GPT2Full(nn.Module):
    """Full GPT-2 model: Embedding + Transformer + LM Head"""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB, HIDDEN)
        self.pos_emb = nn.Embedding(MAX_SEQ, HIDDEN)
        self.blocks = nn.ModuleList([
            TransformerBlock(HIDDEN, HEADS, HEAD_DIM, FF_DIM) for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        # Cast float input_ids to long for embedding lookup
        ids_long = input_ids.long()
        x = self.token_emb(ids_long) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, S, vocab)
        return logits


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Creating GPT-2 full model:")
    print(f"  vocab={VOCAB}, hidden={HIDDEN}, heads={HEADS}, head_dim={HEAD_DIM}")
    print(f"  ff_dim={FF_DIM}, layers={LAYERS}, max_seq={MAX_SEQ}")

    model = GPT2Full()
    model.eval()

    # Create dummy input (token IDs)
    dummy_ids = torch.randint(0, VOCAB, (1, 8))

    onnx_path = os.path.join(TEST_DIR, "gpt2_full.onnx")

    # Export ONNX with opset 17 (supports GELU tanh mode, avoids Erf/Where)
    # Use operator_export_type to force Tanh-based GELU
    torch.onnx.export(
        model,
        dummy_ids,
        onnx_path,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["logits"],
        custom_opsets={},
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    op_types = set(n.op_type for n in onnx_model.graph.node)
    print(f"ONNX ops used ({len(op_types)}): {sorted(op_types)}")

    # Generate test data
    test_ids = torch.tensor([[42, 100, 7, 200, 50, 150, 33, 88]])  # (1, 8)
    print(f"Input token IDs: {test_ids[0].tolist()}")

    # ONNX Runtime reference
    ort_session = ort.InferenceSession(onnx_path)
    ort_outputs = ort_session.run(None, {"input_ids": test_ids.numpy()})
    logits = ort_outputs[0]  # (1, 8, 256)
    print(f"Logits shape: {logits.shape}")

    # Greedy next token prediction
    next_token = int(np.argmax(logits[0, -1, :]))
    print(f"Next token prediction (greedy): {next_token}")

    # Write simple vocabulary file (just index→string mapping)
    vocab_path = os.path.join(TEST_DIR, "gpt2_full_vocab.txt")
    with open(vocab_path, "w") as f:
        for i in range(VOCAB):
            f.write(f"token_{i}\n")
    print(f"Wrote {vocab_path}")

    # Save test input
    input_path = os.path.join(TEST_DIR, "gpt2_full_input.bin")
    with open(input_path, "wb") as f:
        f.write(test_ids.numpy().astype(np.int64).tobytes())
    print(f"Wrote {input_path}")

    # Save reference logits
    ref_path = os.path.join(TEST_DIR, "gpt2_full_logits.bin")
    with open(ref_path, "wb") as f:
        f.write(logits.astype(np.float32).tobytes())
    print(f"Wrote {ref_path} ({logits.nbytes} bytes)")

    # Verify PyTorch vs ONNX Runtime
    with torch.no_grad():
        pt_logits = model(test_ids).numpy()
    max_diff = np.abs(pt_logits - logits).max()
    print(f"PyTorch vs ONNX Runtime max_diff: {max_diff:.2e}")

    print("\nONNX graph nodes:")
    for i, node in enumerate(onnx_model.graph.node):
        inputs_str = ", ".join(node.input[:3])
        if len(node.input) > 3:
            inputs_str += ", ..."
        outputs_str = ", ".join(node.output[:2])
        print(f"  [{i:2d}] {node.op_type:12s}  {inputs_str:60s} → {outputs_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
