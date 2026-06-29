#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import models

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - optional dependency
    smp = None

from semantic_segmentation.transforms import get_transforms
from semantic_segmentation.utils import (
    LABELS_NAMES_MAP,
    MASK_CLASS_COLORS,
    get_prediction,
    load_image,
    prepare_for_prediction,
)


class SegmentationModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output
        return {"out": output}


def build_model(
    num_classes: int,
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    model_type: str = "deeplabv3plus",
):
    if model_type == "deeplabv3plus":
        if smp is not None:
            print("Using DeepLabV3+ with an EfficientNet encoder.")
            model = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                classes=num_classes,
                activation=None,
            )
        else:
            print("segmentation_models_pytorch is not installed; falling back to torchvision DeepLabV3.")
            model = models.segmentation.deeplabv3_resnet101(
                weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
                progress=True,
            )
            model.classifier[4] = torch.nn.Conv2d(
                in_channels=256, out_channels=num_classes, kernel_size=1
            )
            model.aux_classifier[4] = torch.nn.Conv2d(
                in_channels=256, out_channels=num_classes, kernel_size=1
            )
    elif model_type == "deeplabv3":
        print("Using torchvision DeepLabV3.")
        model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
            progress=True,
        )
        model.classifier[4] = torch.nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1
        )
        model.aux_classifier[4] = torch.nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1
        )
    elif model_type == "fcn":
        print("Using torchvision FCN.")
        model = models.segmentation.fcn_resnet101(
            weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT,
            progress=True,
        )
        model.classifier[4] = torch.nn.Conv2d(
            in_channels=2048, out_channels=num_classes, kernel_size=1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = SegmentationModelWrapper(model)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.replace("module.", "", 1): value for key, value in state_dict.items()
            }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def save_outputs(mask: np.ndarray, image_path: Path, output_dir: Path, num_classes: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"

    cv2.imwrite(str(mask_path), mask.astype(np.uint8))

    overlay = MASK_CLASS_COLORS[mask]
    overlay = overlay.astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")

    counts = np.bincount(mask.ravel(), minlength=num_classes)
    print("Class distribution:")
    for class_id, count in enumerate(counts):
        print(f"  {class_id}: {LABELS_NAMES_MAP.get(class_id, str(class_id))} -> {count} pixels")


def main():
    parser = argparse.ArgumentParser(description="Segment a single image with the DeepLabV3 model from this repository")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--checkpoint", default=None, help="Optional path to a .pt or .pth checkpoint")
    parser.add_argument("--output-dir", default="outputs", help="Directory where the mask and overlay will be written")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--num-classes", type=int, default=12, help="Number of output classes")
    parser.add_argument(
        "--model-type",
        default="deeplabv3plus",
        choices=["deeplabv3plus", "deeplabv3", "fcn"],
        help="Segmentation architecture to use",
    )
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(args.device)
    model = build_model(args.num_classes, args.checkpoint, device, model_type=args.model_type)

    _, _, test_transforms = get_transforms(mask_fill_value=0)
    image = load_image(str(image_path))
    prepared = prepare_for_prediction(
        image,
        test_transforms,
        device,
        target_height=args.resize_height,
        target_width=args.resize_width,
    )

    with torch.inference_mode():
        pred = get_prediction(model, prepared)

    mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    save_outputs(mask, image_path, output_dir, args.num_classes)


if __name__ == "__main__":
    main()
