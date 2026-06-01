# Real-Time Face Mask Detection System

This repository provides a complete end-to-end solution for detecting face masks in real-time using **TensorFlow 2** and the **SSD MobileNet V2 FPNLite** architecture.

## 🚀 Features
- **Comprehensive Pipeline**: Covers everything from path setup and TFRecord creation to model training and real-time inference.
- **Optimized Inference**: Implements **Non-Max Suppression (NMS)** to ensure clean, non-overlapping detections.
- **Custom Dataset Integration**: Fine-tuned using transfer learning for high accuracy on mask/no-mask classification.

## 📂 Project Structure
```text
RealTimeObjectDetection/
├── Tensorflow/
│   ├── models/              # TensorFlow Models Garden
│   ├── scripts/             # Conversion and helper scripts
│   └── workspace/
│       ├── annotations/     # Label map and TFRecords
│       ├── images/          # Train and Test image sets
│       ├── models/          # Custom trained models (checkpoints)
│       └── pre-trained-models/ # Base models from TF Model Zoo
│ 
├── Face_Mask_Detection_Setup_ipynb.ipynb
├── RealTimeMaskDetector.ipynb 
├── training_output.log 
├── pbtxt/ 
└── README.md
```

## 📊 Data Source & Annotation
- **Dataset**: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by Omkar Gurav via Kaggle.
- **Labeling Tool**: Images were annotated using [labelImg](https://github.com/tzutalin/labelImg) to generate PascalVOC XML annotations.
- **Classes**:
  1. `Mask` 
  2. `NoMask` 

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Before setting up the project, ensure you have the correct versions of Python and Protobuf installed to prevent compatibility issues with the TensorFlow Object Detection API:
- **Python**: `3.10.x`
- **Protobuf**: `3.19.x`

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/RealTimeObjectDetection.git](https://github.com/your-username/RealTimeObjectDetection.git)
cd RealTimeObjectDetection
```

### 3. Environment Setup
It is highly recommended to use a virtual environment. Create one using Python 3.10, activate it, and install the strictly versioned base dependencies:
```bash
# Create and activate environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Upgrade pip and install matching package versions
python3.10 -m pip install --upgrade pip
python3.10 -m pip install tensorflow==2.15.0 keras==2.15.0 tf_keras opencv-python matplotlib
```

### 4. TensorFlow Object Detection API & Protobuf Compilation
You must have the [TensorFlow Models Garden](https://github.com/tensorflow/models) installed in your `Tensorflow/models` directory. 

Ensure you have **protoc version 3.19** installed on your system to compile the Protobuf files required for the `object_detection` module:

```bash
# Navigate to the research directory
cd Tensorflow/models/research

# Compile protobuf files using protoc 3.19
protoc object_detection/protos/*.proto --python_out=.

# Install the Object Detection API
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

---

## ⚙️ Project Workflow

### Phase 1 & 2: Data Preparation + Training (Setup Notebook)
All preprocessing and model training are done inside:
**📓 `Face_Mask_Detection_Setup.ipynb`**

This includes:
- Dataset labeling and preparation
- Creating `label_map.pbtxt`
- Converting dataset into TFRecord format
- Configuring SSD MobileNet V2 pipeline
- Transfer learning using COCO pretrained weights
- Model training and checkpoint saving

### Phase 3: Real-Time Detection (Detector Notebook)
Real-time inference is handled inside:
**📓 `RealTimeMaskDetector.ipynb`**

This includes:
- Loading trained model checkpoint
- Capturing webcam stream using OpenCV
- Running real-time inference on frames
- Applying Non-Max Suppression (NMS)
- Displaying Mask / No Mask predictions

---

## 🎯 Workflow Summary
```text
Dataset → Setup Notebook (Prep + Training) → Trained Model → Detector Notebook (Real-Time Inference)
```

## 🎮 Controls
- During the live detection session, press **'q'** to safely close the webcam feed.

## 📄 License
This project is licensed under the MIT License.
```
