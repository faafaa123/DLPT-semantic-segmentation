# Semantic Segmentation: Drone aerial views

This is the fourth project of the Opencv University course ["Deep Learning with PyTorch"](https://opencv.org/university/deep-learning-with-pytorch/).
It focuses on applying semantic segmentation on images taken from drones to differentiate between 12 classes.


## Introduction

Semantic segmentation is a task in computer vision, where the objective is to assign a class label to every pixel in an 
image. This project focuses on classifying the pixels of images taken from drones into 12 classes.


## Data

The project uses a dataset of 3269 images of size 1280(W) x 720(H), taken by drones, and annotated image masks for 
the following 12 classes: 

    background, person, bike, car, drone, boat, animal, obstacle, construction, vegetation, road, sky

Examples:
![img.png](media/img.png)

## The method used

Fine-tuning of a DeepLabV3 ResNet-101 pre-trained model using a custom PyTorch training loop. The objective was to learn
how to manually implement all the required steps, particularly the ones of the training loop.

- The dataset was split using a stratified shuffle split scheme into train and validation subsets with 80% and 20% of the 
available data, respectively. The stratification was done based on the presence or not of a class in each image. 

- Various augmentations techniques were used to try to improve generalization.

- The loss function used was an equally weighted combination of the Focal Loss and the Soft Dice Loss:
  - The Focal Loss is a modification of the Cross-Entropy loss focused on learning from hard negative examples.
  - The Soft Dice Loss is effective in addressing the challenge of imbalanced foreground and background regions.

- An SGD optimizer using the setup used by the YOLOv5 training script, where three parameter groups are defined for 
different weight decay configurations.

- A learning rate scheduler that implements the 1-cycle policy. It adjusts the learning rate from an initial rate to a 
maximum, then decreases it to a much lower minimum.

- The custom training loop includes:
    - updating the optimizer learning rate by using a LR scheduler
    - gradient accumulation
    - evaluation on the validation set
    - tracking of training losses and scores
    - tracking of validation losses and scores
    - tracking of per-class scores


## Discussion

The model used is DeeplabV3, trained for 20 epochs, which resulted in a Dice Score of `0.76025` on the Kaggle competition Private Set.
(latest update yet to be pushed).

Further improvements to the data splitting process could incorporate the pixel count for each class in every image, 
so that the images are distributed in a way that considers the occurrence of each class, weighted by the size of 
the objects.

See the [notebook](project-4-deep-learning-with-pytorch-2024.ipynb).