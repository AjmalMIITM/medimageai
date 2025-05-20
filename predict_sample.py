import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('skin_lesion_classifier.h5')

# Class labels (must match your training generator's class order)
class_labels = sorted(os.listdir('preprocessed_isic/Train'))

# Pick a sample image from the test set
img_path = 'preprocessed_isic/Test/melanoma/ISIC_0000002.jpg'  # Change to any test image

# Load and preprocess the image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
pred_class = class_labels[np.argmax(pred)]

print(f"Predicted class: {pred_class}")