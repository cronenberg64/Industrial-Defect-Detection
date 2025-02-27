import pandas as pd

# df = pd.read_csv("documents/Programming_Projects/Industrial-Defect-Detection/data/severstal-steel-defect-detection/train.csv")
df = pd.read_csv("data/severstal-steel-defect-detection/train.csv")
print(df.head())

# Check how many images have defects
num_defective = df["EncodedPixels"].notna().sum()
num_no_defect = df["EncodedPixels"].isna().sum()

print(f"Defective Images: {num_defective}")
print(f"Non-Defective Images: {num_no_defect}")



