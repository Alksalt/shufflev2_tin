# shufflev2_tin/scripts/test_shufflenetv2.py
from __future__ import annotations
import torch
from shufflev2_tin.models import from_config

CFG = {
    "model": {
        "arch": "shufflenetv2",
        "width_mult": 1.0,
        "num_classes": 200,
        "dropout": 0.1,
        "norm": "bn",
        "act": "relu",
        "out_channels": [24, 116, 232, 464, 1024],
        "stages": [4, 8, 4],
        "stem_stride": 2,
        "last_pool": "global_avg",
    }
}

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = from_config(CFG).to(dev)

    # param count
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e6:.3f}M  (expected ~2.2–2.4M)")

    # forward (AMP-safe)
    x = torch.randn(8, 3, 64, 64, device=dev)
    enabled = dev.type == "cuda"
    with torch.autocast(device_type=dev.type, enabled=enabled):
        y = model(x)
    print("Forward:", tuple(y.shape))
    assert y.shape == (8, 200)

    # channels_last
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", enabled=True):
            y = model(x)
        print("Forward (channels_last):", tuple(y.shape))
        assert y.shape == (8, 200)

if __name__ == "__main__":
    main()
