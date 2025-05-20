import cv2
import os

input_base = "Skin cancer ISIC The International Skin Imaging Collaboration"
output_base = "preprocessed_isic"

splits = ["Train", "Test"]
target_size = (224, 224)

for split in splits:
    input_split = os.path.join(input_base, split)
    output_split = os.path.join(output_base, split)
    os.makedirs(output_split, exist_ok=True)
    classes = [d for d in os.listdir(input_split) if os.path.isdir(os.path.join(input_split, d))]
    for cls in classes:
        input_class = os.path.join(input_split, cls)
        output_class = os.path.join(output_split, cls)
        os.makedirs(output_class, exist_ok=True)
        for fname in os.listdir(input_class):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(input_class, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    out_path = os.path.join(output_class, fname)
                    cv2.imwrite(out_path, img_resized)
        print(f"Processed {cls} in {split} split.")
