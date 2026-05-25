"""
Train a small CNN on MNIST and export to ONNX for CudaForge testing.

Architecture (only ops CudaForge supports):
  Input(1,1,28,28) -> Conv(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
  -> Conv(8->16, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
  -> Reshape(1, 16*7*7) -> Gemm(784, 10) -> Softmax
"""
import struct, os, sys, gzip, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTPUT_DIR, "mnist_classifier.onnx")
INPUT_BIN  = os.path.join(OUTPUT_DIR, "mnist_input.bin")
REF_BIN    = os.path.join(OUTPUT_DIR, "mnist_ref_output.bin")

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    result = {}
    for name, url in MNIST_URLS.items():
        fname = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(fname):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, fname)
        with gzip.open(fname, 'rb') as f:
            if 'images' in name:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                result[name] = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            else:
                magic, num = struct.unpack(">II", f.read(8))
                result[name] = np.frombuffer(f.read(), dtype=np.uint8)
    return (result["train_images"], result["train_labels"],
            result["test_images"], result["test_labels"])


class TinyMNIST(nn.Module):
    """Core model — outputs raw logits (no softmax)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.fc = nn.Linear(16 * 7 * 7, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ExportModel(nn.Module):
    """Wrapper that adds Softmax for ONNX export."""
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        x = self.core(x)
        return F.softmax(x, dim=1)


def main():
    device = torch.device("cpu")
    torch.manual_seed(42)

    # Download and prepare MNIST
    train_img, train_lbl, test_img, test_lbl = download_mnist("./mnist_data")
    train_img = (train_img.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    test_img  = (test_img.astype(np.float32) / 255.0 - 0.1307) / 0.3081

    train_t = torch.from_numpy(train_img.copy()).unsqueeze(1)
    test_t  = torch.from_numpy(test_img.copy()).unsqueeze(1)
    train_l = torch.from_numpy(train_lbl.copy()).long()
    test_l  = torch.from_numpy(test_lbl.copy()).long()

    model = TinyMNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    n_train = len(train_t)
    batch_size = 64

    # Training
    for epoch in range(1, 6):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            data, target = train_t[idx].to(device), train_l[idx].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            pred = model(test_t.to(device)).argmax(dim=1)
            acc = (pred == test_l.to(device)).float().mean().item() * 100
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, test_accuracy={acc:.2f}%")

    # Export wrapped model with Softmax
    print("\nExporting to ONNX...")
    export_model = ExportModel(model).to(device)
    export_model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    torch.onnx.export(
        export_model, dummy_input, MODEL_PATH,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
    )

    # PyTorch ≥ 2.12 exports with external data by default.
    # Re-save with inline raw_data so CudaForge's protobuf parser can read weights.
    m = onnx.load(MODEL_PATH)
    onnx.save_model(m, MODEL_PATH, save_as_external_data=False)
    # Clean up stale external data file if it was created
    data_file = MODEL_PATH + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"Model saved: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

    # Compute reference output for first test image
    model.eval()
    img = test_t[0:1]
    label = test_l[0].item()

    with torch.no_grad():
        logits = model(img)
        ref_probs = F.softmax(logits, dim=1).squeeze().numpy()

    img_np = img.numpy().astype(np.float32)

    print(f"\nTest image: digit {label}")
    print(f"Image stats: min={img_np.min():.4f}, max={img_np.max():.4f}, mean={img_np.mean():.4f}")
    print(f"\nReference output (10 classes):")
    pred_cls = int(np.argmax(ref_probs))
    for i in range(10):
        marker = " <-- predicted" if i == pred_cls else ""
        print(f"  Class {i}: {ref_probs[i]:.6f}{marker}")

    with open(INPUT_BIN, "wb") as f:
        f.write(img_np.tobytes())
    with open(REF_BIN, "wb") as f:
        f.write(ref_probs.astype(np.float32).tobytes())
    print(f"\nInput: {INPUT_BIN} ({os.path.getsize(INPUT_BIN)} bytes)")
    print(f"Reference: {REF_BIN} ({os.path.getsize(REF_BIN)} bytes)")

    # Print model ops for verification
    print("\n--- ONNX Model Ops ---")
    import onnx
    m = onnx.load(MODEL_PATH)
    for node in m.graph.node:
        print(f"  {node.op_type}: inputs={list(node.input)} -> outputs={list(node.output)}")
    print(f"\nInitializers:")
    for init in m.graph.initializer:
        print(f"  {init.name}: dims={list(init.dims)}")
    unsupported = set()
    for node in m.graph.node:
        if node.op_type not in ("Conv", "Relu", "MaxPool", "Reshape", "Gemm", "Softmax"):
            unsupported.add(node.op_type)
    if unsupported:
        print(f"\n*** WARNING: Unsupported ops: {unsupported} ***")
    else:
        print(f"\nAll ops supported by CudaForge!")


if __name__ == '__main__':
    main()
