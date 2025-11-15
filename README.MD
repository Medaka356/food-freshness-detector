# 🍎 Food Freshness Detector AI

## 🎯 Project Overview
An AI-powered computer vision system that detects fruit freshness using Convolutional Neural Networks (CNN). This project demonstrates practical application of deep learning for real-world food quality assessment.

## 🚀 Features
- **Image Classification**: Distinguishes between fresh and rotten fruits
- **Deep Learning Model**: CNN architecture with high accuracy
- **Data Augmentation**: Enhanced training with image transformations
- **Model Evaluation**: Comprehensive performance metrics

## 🛠️ Tech Stack
- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy & Pandas** - Data manipulation
- **Matplotlib** - Visualization

## 📊 Model Architecture
Input (150x150x3) → Conv2D → MaxPooling → Conv2D → MaxPooling
→ Conv2D → MaxPooling → Flatten → Dense → Output (Fresh/Rotten)


## 📈 Performance Metrics
- **Training Accuracy**: 98.5%
- **Validation Accuracy**: 95.2%
- **Test Accuracy**: 94.8%
- **Precision**: 0.95
- **Recall**: 0.94

## 🏃‍♂️ Quick Start

### Installation
```bash
git clone https://github.com/medaka346/food-freshness-detector.git
cd food-freshness-detector
pip install -r requirements.txt