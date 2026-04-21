# Task 3: Cats vs Dogs Classification using SVM

This script performs binary image classification to distinguish between cats and dogs using Support Vector Machine (SVM) with OpenCV for image processing.

## Features

- Generates a sample dataset if not present (20 images each for cats and dogs)
- Loads and preprocesses images (resizes to 64x64)
- Trains an SVM model
- Evaluates accuracy on test data

## Requirements

- Python 3.x
- opencv-python
- numpy
- scikit-learn
- pillow

## Installation

Install the required packages via the root `requirements.txt`.

## Usage

Run the script:
```
python task3.py
```

The script will generate sample images if the `dataset` folder doesn't exist, train the model, and print the accuracy.

## Dataset

- Images are in `dataset/cats/` and `dataset/dogs/`.
- If not present, generates simple colored square images (red for cats, blue for dogs).
- Images are resized to 64x64 pixels and flattened for SVM input.

## Output

- Prints the accuracy of the model on the test set.

## Troubleshooting

- Ensure packages are installed.
- The sample dataset is basic; for real classification, replace with actual cat/dog images.