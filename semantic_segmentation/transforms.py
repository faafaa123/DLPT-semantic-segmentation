import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mask_fill_value, should_crop=False, crop_size=(None, None)):

    # TODO:
    # Replace deprecated parameters value, mask_value
    # with fill, fill_mask
    # Albumentations 1.3.1 uses them, new 2.0 release doesn't

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Rotate(
            #     limit=10,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     p=0.1,
            # ),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                rotate_limit=0,  # degrees
                scale_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=mask_fill_value,
                # interpolation=cv2.INTER_CUBIC,
                p=0.5,
            ),
            # A.ElasticTransform(
            #     alpha=60,
            #     sigma=8,
            #     value=0,
            #     mask_value=mask_fill_value,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     interpolation=cv2.INTER_CUBIC,
            #     p=0.1,
            # ),
            # A.GridDistortion(
            #     num_steps=10,
            #     distort_limit=0.35,
            #     value=0,
            #     mask_value=mask_fill_value,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     interpolation=cv2.INTER_CUBIC,
            #     p=0.1,
            # ),
            # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
            # A.ColorJitter(
            #     brightness=(0.5, 1.5),
            #     contrast=(0.5, 1.5),
            #     saturation=(0.5, 1.5),
            #     hue=0.01,  # must be in this interval [-0.5, 0.5].
            #     p=0.1,
            # ),
            # A.PixelDropout(p=0.1),
            # A.CoarseDropout(p=0.1),
            # A.ISONoise(p=0.1),
            # A.Blur(p=0.01),
            # A.MedianBlur(p=0.01),
            # A.ToGray(p=0.01),
            # A.CLAHE(p=0.01),
            # A.RandomGamma(p=0.1),
            # A.ImageCompression(quality_lower=75, p=0.01),
            # A.Perspective(fit_output=True, pad_mode=cv2.BORDER_CONSTANT, p=0.1),
            # A.CropNonEmptyMaskIfExists(),  # TODO: complete this
            A.Normalize(  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255,
            ),
            # HWC format to PyTorch CHW format
            # HW format to PyTorch 1HW format (adds channel dimension)
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose(
        [
            A.Normalize(  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = valid_transforms

    # https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
    # The inference transforms are available at DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms and
    # perform the following preprocessing operations:
    #   - Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
    #   - The images are resized to resize_size=[520] using interpolation=InterpolationMode.BILINEAR.
    #   - The values are first rescaled to [0.0, 1.0]
    #   - then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

    return train_transforms, valid_transforms, test_transforms
