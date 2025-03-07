 # PROJECT DOCUMENTATION
https://docs.google.com/document/d/12ox6uQmXCqGPOZbz22mYYUy25Zze7LfQRF1iKFYL4wI/edit?tab=t.0

# Endangered Animal Detection using YOLOv8  
This project trains a YOLOv8 model to detect different animals from images.  
It uses a custom dataset with 22K images and is trained on Kaggle.  

## Dataset  
- The dataset consists of 22,000 images.  
- It is organized into `train`, `val`, and `test` folders.  
- The dataset annotation format is in YOLO `.txt` format.

## Installation  
Make sure you have the required dependencies installed:  
pip install ultralytics opencv-python torch torchvision

## Training  
Train the model using the following script:

from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")  

# Train the model
model.train(
    data="data.yaml",  # Path to dataset config file
    epochs=50,
    imgsz=640,
    batch=8,
    cache=True,
    project="TrainingResults",
    name="animalDetection"
)

## Inference  
To test the trained model on new images:
python
model = YOLO("best.pt")  # Load trained model
results = model.predict("test_image.jpg", save=True)

## Project Structure  
📂 project_folder/  
 ├── 📂 dataset/  
 │   ├── train/  
 │   ├── valid/  
 │   ├── test/  
 ├── 📂 TrainingResults/  
 │   ├── animalDetection/  
 │   │   ├── weights/  
 │   │   │   ├── best.pt  
 ├── data.yaml  
 ├── train.py  
 ├── test.py  
 ├── README.md 

 ## Issues & Contributions  
Feel free to open issues or contribute by making pull requests.  
