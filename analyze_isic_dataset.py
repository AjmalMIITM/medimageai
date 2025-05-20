import os

def analyze_split(split_dir):
    print(f"\nAnalyzing '{split_dir}' split:")
    split_path = os.path.join(dataset_dir, split_dir)
    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    for cls in classes:
        class_dir = os.path.join(split_path, cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"  {cls}: {len(images)} images")
        print(f"    Sample images: {images[:3]}")

# Set this to your extracted dataset directory name
dataset_dir = "Skin cancer ISIC The International Skin Imaging Collaboration"

# Analyze both Train and Test splits
analyze_split("Train")
analyze_split("Test")