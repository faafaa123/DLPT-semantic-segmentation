# Standard Library imports
from pathlib import Path

# External imports
import pandas as pd
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from pytorch_toolbelt.utils.rle import rle_encode, rle_to_string, rle_decode

# Local imports
from semantic_segmentation.utils import (
    torch_to_cv2,
    denormalize,
    MASK_CLASS_COLORS,
    LABELS_NAMES_MAP,
    get_prediction,
)


def draw_image_mask_prediction(
    image,
    ax,
    mask=None,
    pred=None,
    is_cv_im=False,
    titles=("Image", "Ground truth mask", "Prediction"),
):
    """ """

    if not is_cv_im:
        image = denormalize(image)
        image = torch_to_cv2(image)

    if mask is not None:
        mask = torch_to_cv2(mask, is_mask=True) if not is_cv_im else mask
        mask = MASK_CLASS_COLORS[mask]

    if pred is not None:
        pred = torch_to_cv2(pred, is_mask=True) if not is_cv_im else pred
        pred = MASK_CLASS_COLORS[pred]

    ax[0].imshow(image)
    ax[0].set_xlabel(titles[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    if mask is not None:
        ax[1].imshow(mask)
        ax[1].set_xlabel(titles[1])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    if pred is not None:
        ax[-1].imshow(pred)
        ax[-1].set_xlabel(titles[2])
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])


def draw_batch(dataset, n_samples=3):
    """ """

    fig, axes = plt.subplots(
        nrows=n_samples, ncols=2, sharey=True, figsize=(10, 3 * n_samples)
    )
    for i in range(n_samples):
        image, mask = dataset[i]
        draw_image_mask_prediction(image, axes[i], mask=mask)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def bar_plot(x, y, xlabel="", ylabel="", title="", figsize=(10, 6)):
    """ """

    plt.figure(figsize=figsize)
    plt.bar(x, y, color="skyblue")
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def draw_mask_overlay(
    image, mask, class_id: int, alpha=0.5, color=(0, 0, 1), is_tensor=True
):
    """ """

    if is_tensor:
        image = denormalize(image)
        image = image.permute(1, 2, 0).detach().cpu().numpy()  # CHW -> HWC
        mask = mask.detach().cpu().numpy().astype(np.uint8)

    overlay = np.copy(image)
    overlay[mask == class_id] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=overlay)

    return overlay


def plot(x, ys: list, labels: list, title: str):
    """ """

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(10, 3))
    for i, y in enumerate(ys):
        label = labels[i]
        plt.plot(x, y, label=label, marker="o")

    plt.legend()
    ax.set_xlabel("Epoch")
    ax.grid(which="major", color="black", linestyle="-")
    ax.grid(which="minor", color="gray", linestyle="-", alpha=0.2)
    plt.title(title)

    return fig, ax


def plot_loss_and_score(H, epochs):
    """ """

    x = [i for i in range(1, epochs + 1)]

    # Loss graph
    y = [H["train_loss"], H["valid_loss"]]
    labels = ["Train loss", "Validation loss"]
    fig0, ax0 = plot(x, y, labels, "Loss")
    plt.show()

    # Score graph
    y = [H["train_score"], H["valid_score"]]
    labels = ["Train mean Dice", "Validation mean Dice"]
    fig1, ax1 = plot(x, y, labels, "Mean Dice score")
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.show()


def plot_score_per_class(H, num_classes):
    """ """

    epochs = len(H["per_class_score"])

    x = [i for i in range(1, epochs + 1)]
    ys = list(zip(*H["per_class_score"]))

    fig, axes = plt.subplots(
        nrows=num_classes, ncols=1, sharey=True, sharex=True, figsize=(8, 24)
    )

    for i in range(len(ys)):
        y = ys[i]
        axes[i].plot(x, y, label=f"Class {LABELS_NAMES_MAP[i]}", marker="o")
        axes[i].legend(loc=2)
        axes[i].set_ylim([0, 1])
        axes[i].xaxis.set_tick_params(which="major", length=0)
        axes[i].xaxis.set_major_locator(MultipleLocator(1))
        axes[i].yaxis.set_major_locator(MultipleLocator(1))
        axes[i].yaxis.set_major_locator(MultipleLocator(0.5))
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.1))
        axes[i].grid(which="major", color="black", linestyle="-")
        axes[i].grid(which="minor", color="gray", linestyle="-", alpha=0.2)

    fig.supxlabel("Epoch")
    fig.supylabel("Dice coefficient")

    plt.tight_layout()
    plt.show()


def visualize_classes(dataset, num_classes):
    """ """

    image, mask = dataset[0]
    fig, axes = plt.subplots(
        nrows=np.ceil(num_classes // 3).astype(np.uint8),
        ncols=3,
        sharey=True,
        figsize=(12, 14),
    )
    for i in range(num_classes):
        overlay = draw_mask_overlay(image, mask, i, alpha=0.5, color=(1, 0, 0))

        ax = axes.flatten()[i]
        ax.imshow(overlay)
        ax.set_xlabel(f"Class '{LABELS_NAMES_MAP[i]}'")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def draw_predictions(model, dataset, num_predictions, include_mask, device):
    """ """

    ncols = 3 if include_mask else 2
    fig, ax = plt.subplots(
        nrows=num_predictions,
        ncols=ncols,
        sharey=True,
        figsize=(10, 3 * num_predictions),
    )  # w, h

    model.eval()
    with torch.no_grad():
        for i, element in enumerate(tqdm(dataset)):

            if i == num_predictions:
                break

            if include_mask:
                image, mask = element
            else:
                image, mask = element, None

            image = image.to(device, dtype=torch.float32)
            image = image.unsqueeze(0)
            pred = get_prediction(model, image)
            draw_image_mask_prediction(
                image, ax[i], mask=mask, pred=pred, is_cv_im=False
            )

    plt.tight_layout()
    plt.show()


def visualize_rle_encoding_decoding(
    image: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    titles: tuple[str, ...],
):
    """
    Visually verify that the RLE encoding works by encoding and decoding a mask.
    """

    fig, axes = plt.subplots(
        nrows=num_classes, ncols=3, sharey=True, figsize=(15, 3 * num_classes)
    )

    image = denormalize(image)
    image_numpy = torch_to_cv2(image)
    mask_numpy = torch_to_cv2(mask, is_mask=True)
    for class_id in range(num_classes):

        # Keep mask of only one class
        class_mask = np.zeros_like(mask_numpy)
        class_mask[mask_numpy == class_id] = 1

        # Encode and then decode the class mask
        encoded_mask = rle_encode(class_mask)
        rle_string = rle_to_string(encoded_mask)
        reconstructed_mask = rle_decode(rle_string, mask_numpy.shape, np.uint8)

        draw_image_mask_prediction(
            image_numpy,
            axes[class_id],
            mask=class_mask,
            pred=reconstructed_mask,
            is_cv_im=True,
            titles=titles,
        )
