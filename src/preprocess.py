import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# Define paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
TARGET_SIZE = (256, 256)

# Create directories
def create_dirs():
    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "train"))
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "test"))

# Process binary classification datasets
def process_binary_dataset(dataset_name, train_dir, test_dir):
    print(f"Processing {dataset_name}...")
    train_defect_path = os.path.join(train_dir, "defect")
    train_no_defect_path = os.path.join(train_dir, "no_defect")

    test_path = test_dir  # Test folder contains images without labels

    for class_name in ["defect", "no_defect"]:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, "train", class_name), exist_ok=True)

    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "test"), exist_ok=True)

    # Process train images
    for category in ["defect", "no_defect"]:
        source_dir = os.path.join(train_dir, category)
        dest_dir = os.path.join(PROCESSED_DATA_PATH, "train", category)
        for img_name in tqdm(os.listdir(source_dir), desc=f"Processing {dataset_name} - {category}"):
            img = cv2.imread(os.path.join(source_dir, img_name), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(os.path.join(dest_dir, img_name), img_resized)

    # Process test images
    for img_name in tqdm(os.listdir(test_path), desc=f"Processing {dataset_name} - test"):
        img = cv2.imread(os.path.join(test_path, img_name), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(os.path.join(PROCESSED_DATA_PATH, "test", img_name), img_resized)

# Process multi-class dataset
def process_multiclass_dataset(dataset_name, data_dir):
    print(f"Processing {dataset_name} (multi-class)...")
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(PROCESSED_DATA_PATH, "train", class_name), exist_ok=True)
            for img_name in tqdm(os.listdir(class_path), desc=f"Processing {dataset_name} - {class_name}"):
                img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, TARGET_SIZE)
                cv2.imwrite(os.path.join(PROCESSED_DATA_PATH, "train", class_name, img_name), img_resized)

# Run preprocessing
if __name__ == "__main__":
    create_dirs()
    process_binary_dataset("Severstal", os.path.join(RAW_DATA_PATH, "Severstal/train"), os.path.join(RAW_DATA_PATH, "Severstal/test"))
    process_binary_dataset("KolektorSDD2", os.path.join(RAW_DATA_PATH, "KolektorSDD2/train"), os.path.join(RAW_DATA_PATH, "KolektorSDD2/test"))
    process_multiclass_dataset("GC10-DET", os.path.join(RAW_DATA_PATH, "GC10-DET"))
    
    print("Preprocessing complete!")








