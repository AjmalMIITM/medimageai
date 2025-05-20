import cv2
import os

input_folder = "input_images"
output_folder = "processed_images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, img_resized)
            print(f"Processed: {filename}")
        else:
            print(f"Failed to load: {filename}")
