# Standard Library imports
import os

# External imports
from torch.utils.data import Dataset

# Local imports
from semantic_segmentation.utils import resize_image, resize_mask, load_image, load_mask


class SemanticSegmentationDataset(Dataset):
    """
    Generic Dataset class for semantic segmentation datasets.
    """

    # TODO: Argument to also return the original mask. To be used in with the validation set.

    def __init__(
        self,
        data_path,
        images_folder,
        masks_folder,
        image_ids,
        transforms=None,
        target_height: int | None = None,
        target_width: int | None = None,
    ):
        """
        Args:
            data_path (string): Path to the dataset folder.
            images_folder (string): Name of the folder containing the images.
            masks_folder (string): Name of the folder containing the masks.
            image_ids (list): List of image IDs to include in the dataset.
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """

        self.data_path = data_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_ids = image_ids
        self.target_height = target_height
        self.target_width = target_width
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get image and mask paths
        image_path = os.path.join(self.data_path, self.images_folder, f"{image_id}.jpg")
        mask_path = os.path.join(self.data_path, self.masks_folder, f"{image_id}.png")

        # Load image and mask
        image = load_image(image_path)
        mask = load_mask(mask_path)

        # Resize image and mask
        if self.target_height is not None and self.target_width is not None:
            image = resize_image(image, self.target_width, self.target_height)
            mask = resize_mask(mask, self.target_width, self.target_height)

        if self.transforms is not None:
            if mask is None:
                return self.transforms(image=image)["image"]
            else:
                transformed = self.transforms(image=image, mask=mask)

            return transformed["image"], transformed["mask"]

        return image, mask
