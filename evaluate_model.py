import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained model
model = tf.keras.models.load_model('skin_lesion_classifier.h5')

# Paths
test_dir = 'preprocessed_isic/Test'

# Get class labels from the training directory (sorted)
class_labels = sorted(os.listdir('preprocessed_isic/Train'))

# Prepare test data generator
img_size = 224
batch_size = 16

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Predict on the whole test set
predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# Calculate accuracy
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)

# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 10))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
cm_display.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(f"Test Accuracy: {accuracy:.2f}")
