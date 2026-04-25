Hybrid Deep Learning for Plant Disease Detection

## Overview
This project presents a hybrid deep learning approach for detecting plant diseases from leaf images. The model combines InceptionV3 and EfficientNetB0 to leverage complementary feature extraction capabilities. A confidence-based prediction mechanism is introduced to evaluate the reliability of predictions, and Grad-CAM is used to provide visual explanations. The system is designed to be accurate, efficient, and interpretable, making it suitable for real-world agricultural applications.

## Features
- Hybrid CNN architecture using InceptionV3 and EfficientNetB0
- Parallel feature extraction and feature fusion
- Confidence-based prediction using softmax probabilities
- Threshold-based decision mechanism (0.7) for reliability
- Grad-CAM visualization for interpretability
- Data augmentation for improved generalization
- Fine-tuning for enhanced performance

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn

## Dataset
- Potato leaf image dataset
- Classes: Early Blight, Late Blight, Healthy
- Data split: Training (~70%), Validation (~15%), Test (~15%)

## Methodology
1. Data Preprocessing: Images are resized to 224×224 pixels and normalized. Data augmentation techniques such as rotation, zoom, and flipping are applied.
2. Feature Extraction: InceptionV3 and EfficientNetB0 are used in parallel to extract complementary features.
3. Feature Fusion: Extracted features are combined using concatenation.
4. Classification: A dense layer followed by a softmax layer performs multi-class classification.
5. Confidence-Based Prediction: Confidence is computed using maximum softmax probability (Conf(x) = max P(y = i | x)). A threshold (0.7) determines whether predictions are confident or uncertain.
6. Explainability: Grad-CAM highlights important regions influencing predictions.

## Model Performance
- Hybrid Model: 98.5% accuracy
- InceptionV3: 80.67% accuracy
- EfficientNetB0: 91.33% accuracy
The hybrid model outperforms individual baseline models and improves reliability.

## Project Structure
project/
│── dataset/
│   ├── Train/
│   ├── Valid/
│   └── Test/
│── models/
│── outputs/
│── notebooks/
│── main.py
│── requirements.txt
│── README.md

## Installation
1. git clone <repository-url>
2. cd <project-folder>
3. pip install -r requirements.txt

## Usage
Run: python main.py
Steps:
- Upload an image
- Set confidence threshold
- View prediction, confidence score, and Grad-CAM visualization

## Results
- High classification accuracy with consistent performance
- Confidence-based filtering improves reliability
- Grad-CAM provides visual interpretability
- Robust evaluation using confusion matrix and metrics

## Limitations
- Trained on a specific dataset (potato leaves)
- Performance may vary under different real-world conditions
- Requires validation on diverse datasets

## Future Work
- Extend to multiple crops and diseases
- Deploy as web/mobile application
- Improve robustness with larger datasets
- Integrate real-time monitoring systems

## References
- Szegedy et al., InceptionV3, 2016
- Tan and Le, EfficientNet, 2019
- Mohanty et al., Plant Disease Detection, 2016
- Alharbi et al., Potato Disease Detection, 2025
- Selvaraju et al., Grad-CAM, 2017
