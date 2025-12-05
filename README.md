# Image Classification using HOG, LBP and SVM

A comprehensive comparative analysis of traditional computer vision feature extraction techniques (HOG vs LBP) for multi-class image classification using Support Vector Machines.


## 📊 Project Overview

This project implements and compares two classical computer vision feature extraction methods:
- **HOG** (Histogram of Oriented Gradients)
- **LBP** (Local Binary Patterns) 

for image classification on a 5-class dataset (City, Face, Green, Office, Sea) using Support Vector Machines (SVM).

## 🎯 Key Features

- **Feature Extraction**: Implementation of HOG and multi-channel LBP feature extractors
- **Machine Learning**: SVM classification with RBF and Sigmoid kernels
- **Comprehensive Evaluation**: 5-fold cross-validation, multiple metrics, confusion matrices
- **Visualization**: Feature maps, performance comparisons, training history
- **Modular Design**: Clean, reusable code structure

## 📈 Results Preview

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| HOG + SVM | 0.85 | 0.86 | 0.85 | 0.85 |
| LBP + SVM | 0.78 | 0.79 | 0.78 | 0.78 |

*HOG features significantly outperform LBP for this image classification task.*

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/salahBerr0/Image-Classification-HOG-LBP-SVM.git
cd image-classification-hog-lbp-svm

#Google Colab code:
 https://colab.research.google.com/drive/1RCfvZOg7B_wpdObyz-yMnKQv60FEtFja?usp=sharing

# Install dependencies
pip install -r requirements.txt

