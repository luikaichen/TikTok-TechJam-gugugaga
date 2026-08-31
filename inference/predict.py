from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "generalisation_model_v2.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


def load_model(checkpoint: Path, device: torch.device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model


def predict_one(
    model,
    image_path: Path,
    device: torch.device,
):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Training / evaluation contract:
    # class 0 = FAKE / AIGC
    # class 1 = REAL
    ai_probability = float(probabilities[0, 0].item())

    return ai_probability


def collect_images(inputs):
    supported = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    paths = []

    for raw in inputs:
        path = Path(raw)

        if path.is_file():
            if path.suffix.lower() in supported:
                paths.append(path)
            continue

        if path.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in supported
                )
            )
            continue

        raise FileNotFoundError(f"Input not found: {path}")

    if not paths:
        raise ValueError("No supported images found.")

    return paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TikTok TechJam Track 5 — "
            "Generalisation V2 inference"
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files and/or directories.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the frozen V2 checkpoint.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional binary demo threshold. "
            "Raw P(AI) is always returned. "
            "Use 0.05 only for the calibrated demo operating point."
        ),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.checkpoint, device)
    images = collect_images(args.inputs)

    for image_path in images:
        ai_probability = predict_one(
            model,
            image_path,
            device,
        )

        result = {
            "image": str(image_path),
            "ai_probability": ai_probability,
        }

        if args.threshold is not None:
            result["threshold"] = args.threshold
            result["demo_label"] = (
                "AIGC"
                if ai_probability >= args.threshold
                else "REAL"
            )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
