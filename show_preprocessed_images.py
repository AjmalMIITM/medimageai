import os
import cv2
import matplotlib.pyplot as plt

base_dir = "preprocessed_isic/Train"
classes = os.listdir(base_dir)

for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if images:
        img_path = os.path.join(class_dir, images[0])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{cls}: {images[0]}")
        plt.axis("off")
        plt.show()
        break  # Remove this break if you want to see one image from each class