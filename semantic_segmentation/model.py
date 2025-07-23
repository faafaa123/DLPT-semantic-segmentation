# External imports
import torch
from torchvision import models
import segmentation_models_pytorch as smp
from torchinfo import summary
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)


def make_deeplabv3_resnet101(num_classes):
    """
    "Dilated convolution" is also called "Atrous convolution"
    """

    # All the layers are unfrozen by default.
    model = models.segmentation.deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True
    )

    print("The model is loaded in eval mode.")
    model.eval()

    print("The model has all its layers unfrozen.")

    # Fresh new last layers of the classifiers
    model.classifier[4] = torch.nn.Conv2d(
        in_channels=256, out_channels=num_classes, kernel_size=1
    )
    model.aux_classifier[4] = torch.nn.Conv2d(
        in_channels=256, out_channels=num_classes, kernel_size=1
    )

    return model


def unfreeze_deeplabv3_resnet101(model, layers: list[str] | None = None):
    """
    Unfreeze only the DeeplabV3 backbone. This won't unfreeze the heads.
    """

    resnet101_layers = ["layer1", "layer2", "layer3", "layer4"]

    # Unfreeze the whole Resnet backbone
    if layers is None:
        print(f"Unfreezing the whole backbone")
        for param in model.backbone.parameters():
            param.requires_grad = True
    else:  # Unfreeze only certain specific layers of the backbone
        for layer in layers:
            assert layer in resnet101_layers, f"Layer '{layer}' isn't part of Resnet101"
            print(f"Unfreezing parameters of the Resnet101 layer '{layer}'")
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True


def unfreeze_deeplabv3_heads(model, all=False):
    """
    DeeplabV3 has the following modules:

    ===========================================================================
    Layer (type:depth-idx)                             Param #
    ===========================================================================
    ├─DeepLabHead: 1-2                                 --
    │    └─ASPP: 2-9                                   --
    │    │    └─ModuleList: 3-34                       15,206,912
    │    │    └─Sequential: 3-35                       328,192
    │    └─Conv2d: 2-10                                589,824
    │    └─BatchNorm2d: 2-11                           512
    │    └─ReLU: 2-12                                  --
    │    └─Conv2d: 2-13                                3,084     <------- Replaced by a new one
    ├─FCNHead: 1-3                                     --
    │    └─Conv2d: 2-14                                (2,359,296)
    │    └─BatchNorm2d: 2-15                           (512)
    │    └─ReLU: 2-16                                  --
    │    └─Dropout: 2-17                               --
    │    └─Conv2d: 2-18                                3,084     <------- Replaced by a new one

    I set the last Conv2d layers are trainable by default, but I want to allow to unfreeze further layers gradually.
    """

    if all:  # Unfreeze the whole head
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for cname, c in model.classifier.named_children():
            print("Children ", cname)
            for p in c.parameters():
                p.requires_grad = True
            print(c)

    print("----------------------")
    for name, p in model.classifier.named_parameters():
        print(name, p.requires_grad)


if __name__ == "__main__":
    model = make_deeplabv3_resnet101(12, torch.device("cpu"))
    unfreeze_deeplabv3_heads(model)

    summary(
        model,
        col_names=[
            "num_params",
            # "params_percent",
            # "kernel_size",
            # "trainable",
        ],
    )
