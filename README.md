# Industrial Defect Detection

## 📌 Project Overview

This project focuses on developing an AI-powered defect detection system for industrial manufacturing. Using computer vision and deep learning, the model aims to identify and classify surface defects in industrial materials, helping to improve quality control and reduce production errors.

### **Key Features:**

✅ Automated defect detection using CNNs and deep learning models\
✅ Supports multiple datasets (Severstal Steel Defect Detection, KolektorSDD2, GC10-DET)\
✅ Preprocessing pipeline for reproducibility and scalability\
✅ Data augmentation techniques to improve model robustness\
✅ Open-source and easy to use for industrial applications

---

## 📂 Repository Structure

```
/industrial-defect-detection
│── /data               # Raw and preprocessed datasets
│── /notebooks          # Jupyter notebooks for analysis and visualization
│── /src                # Source code for training and inference
│    │── preprocess.py  # Script for preprocessing datasets
│    │── train.py       # Model training script
│    │── inference.py   # Inference script for testing new images
│── /models             # Trained models
│── README.md           # Project documentation
```

---

## 📊 Datasets

This project uses publicly available datasets for defect detection:

- **Severstal Steel Defect Detection** - Surface defect detection in steel manufacturing.
- **KolektorSDD2** - Anomalous defect detection for small-scale industrial parts.
- **GC10-DET** - General defect detection for industrial components.

---

## 🔧 Setup & Installation

### **1️⃣ Clone the Repository**

```bash
$ git clone https://github.com/yourusername/industrial-defect-detection.git
$ cd industrial-defect-detection
```

### **2️⃣ Install Dependencies**

```bash
$ pip install -r requirements.txt
```

### **3️⃣ Preprocess the Dataset**

```bash
$ python src/preprocess.py
```

### **4️⃣ Train the Model**

```bash
$ python src/train.py
```

### **5️⃣ Run Inference**

```bash
$ python src/inference.py --image path/to/image.jpg
```

