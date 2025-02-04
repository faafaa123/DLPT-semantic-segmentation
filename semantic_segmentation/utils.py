# Standard Library imports
import os
import gc
from collections import defaultdict
from pathlib import Path
import random

# External imports
import cv2
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch_lr_finder import LRFinder
from torch.optim.optimizer import Optimizer
from pytorch_toolbelt.utils.rle import rle_encode, rle_to_string
import torchvision.transforms.functional as F

# Local imports
from semantic_segmentation.configuration import SystemConfig


# Create colors for the visualization, one for each class
MASK_CLASS_COLORS = np.array(
    [
        [0, 0, 0],  # background
        [192, 128, 128],  # person
        [0, 128, 1],  # bike
        [128, 128, 128],  # car
        [128, 0, 0],  # drone
        [1, 0, 128],  # boat
        [193, 0, 129],  # animal
        [192, 0, 0],  # obstacle
        [192, 129, 0],  # construction
        [0, 65, 1],  # vegetation
        [127, 128, 0],  # road
        [0, 128, 129],  # sky
    ]
)

LABELS_NAMES_MAP = {
    0: "background",
    1: "person",
    2: "bike",
    3: "car",
    4: "drone",
    5: "boat",
    6: "animal",
    7: "obstacle",
    8: "construction",
    9: "vegetation",
    10: "road",
    11: "sky",
}


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()
        gc.collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        gc.collect()


def torch_to_cv2(image: torch.Tensor, is_mask=False) -> np.ndarray:
    """
    Convert a PyTorch image tensor to an OpenCV image.
    """
    image = image.detach()

    if is_mask:
        if image.ndim == 3:
            image = image.squeeze()
        image = image.to(torch.uint8)

    else:
        if image.ndim == 4:
            image = image.squeeze()
        image = image.permute(1, 2, 0)

    return image.cpu().numpy()


def extract_and_onehot_encode_classes_from_multilabel_masks(dataset, num_classes):
    """
    Extract the classes available in each multilabel binary mask and one-hot encodes them.

    Returns:
        A NumPy array of one-hot encoded classes, where the dimensions are
        (num_images, num_classes). Each value in the array represents the presence
        (1) or absence (0) of a specific class in the corresponding image.
    """
    ys = []
    for i in tqdm(range(len(dataset))):
        image, mask = dataset[i]
        y = np.unique(mask).reshape(1, -1)
        y = torch.Tensor(y).to(torch.int64)
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = y.sum(axis=1)
        y = y.numpy()
        ys.append(y)

    return np.concatenate(ys, axis=0)


def denormalize(tensors):
    """
    Denormalize image tensors back to range [0.0, 1.0]

    Modified from: Deep Learning with PyTorch - OpenCV University
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    for c in range(3):
        if len(tensors.shape) == 4:
            tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
        elif len(tensors.shape) == 3:
            tensors[c, :, :].mul_(std[c]).add_(mean[c])
        else:
            raise Exception(
                "Can only deal with images of shape (N, C, H, W) or (C, H, W)"
            )

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


def find_best_lr(
    model,
    loss_fun,
    dataloader,
    grad_accum_steps,
    device,
    start_lr=1e-7,
    end_lr=1,
    num_iter=200,
    momentum=None,
):
    """ """
    temp_optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=momentum)

    lr_finder = LRFinder(model, temp_optimizer, loss_fun, device=device)
    lr_finder.range_test(
        dataloader,
        end_lr=end_lr,
        num_iter=num_iter,
        accumulation_steps=grad_accum_steps,
    )
    lr_finder.plot()
    lr_finder.reset()

    best_lr = extract_best_lr(lr_finder)
    return best_lr


def extract_best_lr(lr_finder):
    """
    Extract the best Learning Rate for a trained LRFinder object.
    """

    learning_rates = np.array(lr_finder.history["lr"])
    losses = np.array(lr_finder.history["loss"])

    min_grad_idx = None
    try:
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
    except ValueError:
        print("Failed to compute the gradients, there might not be enough points.")

    if min_grad_idx is not None:
        best_lr = learning_rates[min_grad_idx]

    return best_lr


def calculate_class_weights(pixel_count_per_class):
    """
    Prepare class weights for BCE loss based on pixel counts, giving higher importance to classes
    with fewer pixels.

        loss_fun = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).to(torch.float32)
        )

    """

    total_pixels = np.sum(pixel_count_per_class)
    pixel_proportion_per_class = pixel_count_per_class / total_pixels

    # Calculate the inverse of the pixel proportions
    class_weights = 1.0 / (
        pixel_proportion_per_class + 1e-12
    )  # Adding epsilon to avoid division by zero

    # Normalize the class weights to make them sum to 1
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


def count_pixels_per_class(images_ids, datapath, num_classes):
    """ """

    pixel_count_per_class = np.zeros(num_classes)

    for image_id in tqdm(images_ids):
        mask_path = os.path.join(datapath, "masks/masks", f"{image_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for i in range(num_classes):
            class_pixels = np.sum(mask == i)
            pixel_count_per_class[i] += class_pixels

    return pixel_count_per_class


def count_images_per_class(dataset):
    """ """

    d = defaultdict(int)
    for i in tqdm(range(len(dataset))):
        image, mask = dataset[i]
        classes = np.unique(mask)
        for c in classes:
            d[c] += 1

    return d


def resize_image(mask, width, height, is_tensor=False):
    if is_tensor:
        F.resize(mask, [height, width], interpolation=F.InterpolationMode.BILINEAR)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, width: int, height: int, is_tensor: bool = False):
    if is_tensor:

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
        return F.resize(
            mask, [height, width], interpolation=F.InterpolationMode.NEAREST
        )

    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def load_checkpoint(model, optimizer, input_path: str, filename: str | None, scaler:torch.amp.grad_scaler.GradScaler):
    """
    Load a saved checkpoint containing model and optimizer states.

    Args:
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into
        input_path: Directory path containing the checkpoint
        filename: Name of the checkpoint file. If None, no loading occurs
        device: Device to load the model to. If None, uses the current device

    Returns:
        tuple containing:
            - epoch: The epoch number when checkpoint was saved
            - loss: The loss value when checkpoint was saved

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is corrupted or incompatible
        ValueError: If input arguments are invalid
    """

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a PyTorch nn.Module")

    if not isinstance(optimizer, Optimizer):
        raise ValueError("optimizer must be a PyTorch Optimizer")

    if filename is None:
        print("No weights were loaded because no checkpoint name was provided.")
        return

    checkpoint_path = Path(input_path, filename)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load saved model and optimizer parameters
    # Load the state dict on the CPU. If the state was saved on the GPU, when reloaded, PyTorch places it back on GPU.
    # https://github.com/huggingface/accelerate/issues/296#issuecomment-1082184342
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    print(f"Successfully loaded checkpoint from '{checkpoint_path}'")
    print(f"Checkpoint was saved at epoch {checkpoint['epoch']+1}.")  # zero-indexed

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded model weights.")

    # Load optimizer weights
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Successfully loaded optimizer weights.")

    if "scaler" in checkpoint and scaler.is_enabled:
        # Resume Amp-enabled runs with bitwise accuracy
        scaler.load_state_dict(checkpoint["scaler"])
        print(f"Successfully loaded scaler.")
    else:
        print(f"No scaler has been loaded. {'Scaler is disabled. ' if not scaler.is_enabled() else ''} {'No scaler available in checkpoint.' if not 'scaler' in checkpoint else ''}")



def create_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """
    Create a new versioned checkpoint directory.

    Args:
        checkpoint_dir: Base directory for checkpoints

    Returns:
        Path: Path pointing to the newly created version directory

    Raises:
        PermissionError: If directory cannot be created due to permissions
    """
    base_dir = Path(checkpoint_dir)

    # Find highest existing version number
    version_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("version_")
    ]
    version_numbers = [
        int(d.name.split("_")[-1])
        for d in version_dirs
        if d.name.split("_")[-1].isdigit()
    ]
    version_num = max(version_numbers, default=-1) + 1

    # Create new version directory
    version_dir = base_dir / f"version_{version_num}"
    version_dir.mkdir(parents=True, exist_ok=False)

    print(f"Checkpoint directory: '{version_dir}'")
    return version_dir


def setup_system(system_config: SystemConfig) -> None:
    """From OpenCV University"""

    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn.benchmark = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


def prepare_for_prediction(
    image, transforms, device, target_height=None, target_width=None
):
    """ """
    if target_height and target_width:
        image = resize_image(image, width=target_width, height=target_height)
    image = transforms(image=image)["image"]
    image = image.to(device, dtype=torch.float32)
    # Before: shape (C, H, W)
    image = image.unsqueeze(0)
    # After: shape (1, C, H, W)
    return image


def get_prediction(model, image):
    """ """
    pred = model(image)["out"]
    pred = pred.argmax(dim=1)  # Collapse from N,C,H,W into N,H,W
    return pred


def create_submission(model, test_ids, test_transforms, config):
    """ """

    output_lines = ["ImageID,EncodedPixels"]

    model.eval()
    with torch.no_grad():
        for image_id in tqdm(test_ids.tolist()):
            image_path = os.path.join(config.DATA_PATH, "imgs/imgs", f"{image_id}.jpg")
            image = load_image(image_path)

            image = prepare_for_prediction(
                image,
                test_transforms,
                config.DEVICE,
                config.RESIZE_HEIGHT,
                config.RESIZE_WIDTH,
            )
            pred = get_prediction(model, image)
            pred = torch_to_cv2(pred)

            # Upscale prediction back to its original size
            pred = resize_mask(pred, config.ORIGINAL_WIDTH, config.ORIGINAL_HEIGHT)

            for class_id in range(config.NUM_CLASSES):
                class_image = np.zeros_like(pred)
                class_image[pred == class_id] = 1
                # NOTE: rle_encode transposes the image internally!
                # I transpose it here to cancel that subsequent transposition.
                pred_rle = rle_to_string(rle_encode(class_image.T))
                output_line = f"{image_id}_{class_id},{pred_rle}"
                output_lines.append(output_line)

    with open(os.path.join(config.OUTPUT_PATH, config.SUBMISSION_FILENAME), "w") as f:
        out = "\n".join(line.strip() for line in output_lines)
        f.write(out)


def inference(
    model,
    scorer,
    valid_ids,
    transforms,
    config,
    upscale_prediction=False,
    downscale_mask=False,
):
    """
    If a resize size is passed, there are two possibilities:
        - score on downscaled image and downscaled mask (gotta downscale the mask, the downscale of the image happens automatically)
        - score on upscaled image and original mask (gotta upscale the prediction, the mask is loaded in its original size by default)

    if no resize size is passed, there is one possibility:
        - score on original image and original mask
    """

    if config.RESIZE_HEIGHT is None and config.RESIZE_WIDTH is None:
        assert not upscale_prediction and not downscale_mask

    if upscale_prediction or downscale_mask:
        assert upscale_prediction != downscale_mask
        assert config.RESIZE_WIDTH is not None and config.RESIZE_HEIGHT is not None

    if upscale_prediction:
        print(
            f"Predictions will be upscaled from {config.RESIZE_HEIGHT}x{config.RESIZE_WIDTH} (HxW) back to their original size of {config.ORIGINAL_HEIGHT}x{config.ORIGINAL_WIDTH} (HxW)"
        )

    if downscale_mask:
        print(
            f"Masks will be downscaled from their original size of {config.ORIGINAL_HEIGHT}x{config.ORIGINAL_WIDTH} (HxW) to {config.RESIZE_HEIGHT}x{config.RESIZE_WIDTH} (HxW)"
        )

    scores = []
    model.eval()
    with torch.no_grad():
        for image_id in tqdm(valid_ids.tolist()):

            ############################################################################################################
            # Prepare image
            ############################################################################################################

            image_path = os.path.join(config.DATA_PATH, "imgs/imgs", f"{image_id}.jpg")
            image = load_image(image_path)

            image = prepare_for_prediction(
                image,
                transforms,
                config.DEVICE,
                config.RESIZE_HEIGHT,
                config.RESIZE_WIDTH,
            )

            with torch.autocast(device_type=str(config.DEVICE), dtype=torch.float16):
                y_pred = model(image)["out"]
                y_pred = y_pred.detach()

            # y_pred shape: (N,C,H,W)
            if upscale_prediction:
                y_pred = resize_mask(
                    y_pred,
                    config.ORIGINAL_WIDTH,
                    config.ORIGINAL_HEIGHT,
                    is_tensor=True,
                )

            y_pred = y_pred.argmax(dim=1)  # Collapse from N,C,H,W into N,H,W

            ############################################################################################################
            # Prepare mask
            ############################################################################################################

            mask_path = os.path.join(config.DATA_PATH, "masks/masks", f"{image_id}.png")

            # Load the mask at its original size
            dummy_image = np.zeros((config.ORIGINAL_HEIGHT, config.ORIGINAL_WIDTH, 3))
            y_true = load_mask(mask_path)
            # Can't pass only mask, gotta pass the image as well
            y_true = transforms(image=dummy_image, mask=y_true)["mask"]

            # y_true shape before:  (H,W)
            if downscale_mask:

                y_true = resize_mask(
                    y_true, config.RESIZE_WIDTH, config.RESIZE_HEIGHT, is_tensor=True
                )
                # y_true shape after:  (1, H, W)

            # If downscale_mask is False, shape will be H,W
            # I don't really need this. I only want to maintain a consistent shape of C,H,W.
            if y_true.ndim == 2:
                y_true = y_true.unsqueeze(0)

            y_true = y_true.to(config.DEVICE, dtype=torch.long)

            ############################################################################################################
            # Score
            ############################################################################################################
            # torch.Size([1, 720, 1280]) torch.Size([720, 1280])
            score = scorer(y_pred, y_true)
            scores.append(score.mean())

    return scores
