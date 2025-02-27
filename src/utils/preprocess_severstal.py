import os
import shutil
import pandas as pd
from pathlib import Path

def preprocess_severstal(data_dir):
    """
    Preprocess Severstal dataset by sorting images into 'defect' and 'no_defect' folders.
    """
    # Define paths
    raw_images_dir = Path(data_dir) / "severstal" / "raw" / "train_images"
    csv_path = Path(data_dir) / "raw" / "severstal-steel-defect-detection" / "train.csv"
    defect_dir = Path(data_dir) / "severstal" / "raw" / "train" / "defect"
    no_defect_dir = Path(data_dir) / "severstal" / "raw" / "train" / "no_defect"
    
    # Ensure directories exist
    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Group by image names (some images have multiple defects)
    defect_images = set(df[df['EncodedPixels'].notna()]['ImageId'])
    
    # Process images
    for img_path in raw_images_dir.glob("*.jpg"):  # Assuming images are .jpg
        if img_path.name in defect_images:
            shutil.move(str(img_path), str(defect_dir / img_path.name))
        else:
            shutil.move(str(img_path), str(no_defect_dir / img_path.name))
    
    print(f"Moved {len(defect_images)} defective images and {len(list(raw_images_dir.glob('*.jpg')))} non-defective images.")

if __name__ == "__main__":
    preprocess_severstal("data")
