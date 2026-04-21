# Task 4: Hand Gesture Recognition using CNN

This script performs image classification for hand gesture recognition using a Convolutional Neural Network (CNN) built with PyTorch.

## Features

- Generates a sample dataset if not present (20 images each for cats and dogs, as placeholders for gestures)
- Loads images from a `dataset` folder with subfolders for each class
- Preprocesses images (resize, normalize)
- Defines and trains a simple CNN model
- Saves the trained model

## Requirements

- Python 3.x
- torch
- torchvision

## Installation

Install the required packages via the root `requirements.txt`.

## Usage

Run the script:
```
python task4.py
```

The script will generate sample images if the `dataset` folder doesn't exist, train the model for 5 epochs, and save it as `gesture_model.pth`.

## Dataset

- Images should be in `dataset/class_name/` folders.
- Generates simple colored square images (red for cats, blue for dogs) as placeholders.
- Images are resized to 64x64 and normalized.

## Output

- Prints training loss for each epoch.
- Saves the model weights to `gesture_model.pth`.

## Troubleshooting

- Ensure packages are installed.
- For real gesture recognition, replace the sample images with actual gesture images in appropriate folders.