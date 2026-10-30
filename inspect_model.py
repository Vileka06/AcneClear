from pathlib import Path
import sys

import torch

# Compatibility with some BasicSR versions that still import this old
# torchvision module path.
try:
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
except Exception:
    pass

from gfpgan import GFPGANer

MODEL_PATH = Path("models/GFPGANv1.3.pth")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Place GFPGANv1.3.pth inside the project's models/ folder."
        )

    print("=" * 72)
    print("ACNECLEAR AI - GFPGAN MODEL INSPECTION")
    print("=" * 72)
    print("Model: GFPGAN v1.3 (pretrained)")
    print("Architecture option used by app.py: clean")
    print("Channel multiplier: 2")
    print("Inference device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print(f"Checkpoint: {MODEL_PATH.resolve()}")
    print(f"Checkpoint size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    restorer = GFPGANer(
        model_path=str(MODEL_PATH),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    generator = restorer.gfpgan

    print("\n" + "=" * 72)
    print("TOP-LEVEL GENERATOR COMPONENTS")
    print("=" * 72)
    for name, module in generator.named_children():
        count = sum(p.numel() for p in module.parameters())
        print(f"{name:24s} {module.__class__.__name__:28s} parameters={count:,}")

    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(
        p.numel() for p in generator.parameters() if p.requires_grad
    )
    print("\n" + "=" * 72)
    print("PARAMETER SUMMARY")
    print("=" * 72)
    print(f"Total generator parameters:     {total_params:,}")
    print(f"Trainable generator parameters: {trainable_params:,}")

    print("\n" + "=" * 72)
    print("FULL GENERATOR ARCHITECTURE")
    print("=" * 72)
    print(generator)

    architecture_file = Path("gfpgan_generator_architecture.txt")
    architecture_file.write_text(str(generator), encoding="utf-8")
    print(f"\nFull architecture saved to: {architecture_file.resolve()}")

    print("\n" + "=" * 72)
    print("IMPORTANT TRAINING INFORMATION")
    print("=" * 72)
    print("This project loads a pretrained GFPGAN v1.3 checkpoint.")
    print("This app does not train GFPGAN from scratch.")
    print("Therefore, this project has no custom training epoch count or")
    print("custom generator/discriminator loss curves to report.")
    print("The app uses the pretrained restoration generator for inference.")
    print("The additional skin softening is a separate OpenCV image-processing step.")

if __name__ == "__main__":
    main()
