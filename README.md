# YOLOv10 Custom Object Detection for Vehicle Detection

Kaggle: www.kaggle.com/ankitparwatkar

This repository contains a custom implementation of YOLOv10 for object detection focused on detecting 5 vehicle classes: Ambulance, Bus, Car, Motorcycle, and Truck. The project includes dataset preparation, visualization, model training, and inference pipelines.

## Dataset Structure
The dataset follows the standard YOLO format:
```
├── data.yaml
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
```

### data.yaml Contents
```yaml
path: ../Cars Detection
train: train/images
val: valid/images

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

## Key Features
- Dataset visualization and annotation tools
- Custom YOLOv10 implementation
- Training and evaluation pipelines
- Visualization of detection results
- Support for Kaggle datasets

## Requirements
- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy
- Matplotlib
- Ultralytics (for YOLOv10)
- Kaggle API (for dataset download)

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib ultralytics kaggle
```

## Getting Started

### 1. Dataset Preparation
Download the dataset from Kaggle and organize it according to the structure above.

### 2. Data Visualization
The notebook includes functions to visualize dataset annotations:

```python
# Visualize sample images with bounding boxes
plot(image_paths='/kaggle/input/cars-detection/Cars Detection/train/images/*',
     label_paths='/kaggle/input/cars-detection/Cars Detection/train/labels/*',
     num_samples=4)
```

### 3. Annotation Conversion
Convert YOLO format annotations to bounding box coordinates:

```python
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1] - bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1] + bboxes[3]/2
    return xmin, ymin, xmax, ymax
```

### 4. Model Training
Train YOLOv10 on the custom dataset:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov10n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov10_vehicle_detection'
)
```

### 5. Inference
Run object detection on new images:

```python
results = model.predict('test.jpg', save=True, conf=0.5)
```

## Results
Performance metrics on validation set:

| Metric    | Value   |
|-----------|---------|
| mAP@0.5   | 0.892   |
| Precision | 0.874   |
| Recall    | 0.856   |
| F1 Score  | 0.865   |


## Repository Structure
```
├── notebooks/                  # Jupyter notebooks
│   ├── computer-vision-yolov10.ipynb
├── data/                       # Dataset
├── src/                        # Source code
│   ├── utils.py                # Utility functions
│   ├── dataset.py              # Dataset handling
│   ├── train.py                # Training script
│   └── detect.py               # Inference script
├── runs/                       # Training results
├── README.md
└── requirements.txt
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License
This project is licensed under the MIT License 

## Acknowledgements
- Ultralytics for YOLOv10 implementation
- Kaggle for hosting the dataset
- COCO dataset for pretrained weights
