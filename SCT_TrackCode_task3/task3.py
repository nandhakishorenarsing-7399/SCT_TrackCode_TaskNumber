import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Generate sample dataset if not exists
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(os.path.join(dataset_dir, 'cats'))
    os.makedirs(os.path.join(dataset_dir, 'dogs'))

    # Generate 20 images for cats (red squares)
    for i in range(20):
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))  # Red
        img.save(os.path.join(dataset_dir, 'cats', f'cat_{i}.jpg'))

    # Generate 20 images for dogs (blue squares)
    for i in range(20):
        img = Image.new('RGB', (64, 64), color=(0, 0, 255))  # Blue
        img.save(os.path.join(dataset_dir, 'dogs', f'dog_{i}.jpg'))

data = []
labels = []

# Load dataset
for category in ['cats', 'dogs']:
    path = os.path.join('dataset', category)
    label = 0 if category == 'cats' else 1

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        data.append(image.flatten())
        labels.append(label)

X = np.array(data)
y = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))