# 🌸 Iris Flower Classification using Logistic Regression

A machine learning project that predicts the species of Iris flowers using Logistic Regression. This project uses the classic Iris dataset to classify flowers into three species: Setosa, Versicolor, and Virginica.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)

## 🔍 Overview

This project demonstrates the implementation of a Logistic Regression classifier to predict Iris flower species based on four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model achieves perfect accuracy in classifying the three Iris species, making it an excellent introductory project for machine learning beginners.

**Developed using Google Colab** - No local setup required! You can run this project directly in your browser.

## 📊 Dataset

The project uses the famous [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from the UCI Machine Learning Repository. 

**Dataset Characteristics:**

Total Samples: 150
Features: 4 (all numeric)
Classes: 3 (Setosa, Versicolor, Virginica)
Samples per Class: 50


**Features:**

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)


## 🛠️ Technologies Used

- Google Colab (Recommended - Cloud-based Jupyter notebook environment)
- Python 3.x
- NumPy (Numerical computations)
- Pandas (Data manipulation and analysis)
- Matplotlib (Data visualization)
- Seaborn (Statistical visualizations)
- scikit-learn (Machine learning implementation)


## 📦 Installation

### Recommended: Run in Google Colab

**No installation needed!** Simply click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/iris-flower-classification/blob/main/Iris_flower_project.ipynb)

**All required libraries are pre-installed in Google Colab.** This is the easiest and recommended way to run this project!

## 🚀 Usage

### Running in Google Colab

1. Click the "Open in Colab" badge above
2. Sign in with your Google account
3. Click Runtime → Run all to execute all cells
4. The notebook will:
   - Load and explore the dataset
   - Visualize the data
   - Train the Logistic Regression model
   - Evaluate model performance
   - Make predictions

## 📈 Model Performance

The Logistic Regression model demonstrates perfect performance on the Iris dataset:

Accuracy: 100%
Precision: 1.00 across all three classes
Recall: 1.00 across all three classes
F1-Score: 1.00 (perfect balance)

### Classification Report

                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00         7
Iris-versicolor       1.00      1.00      1.00        12
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30


## 🎯 Results

### Key Findings

**1. Feature Importance:**

- Petal length and petal width are the most discriminative features
- Sepal measurements provide additional classification support

**2. Class Separability:**

- Iris Setosa is linearly separable from other species
- Versicolor and Virginica are well-distinguished by the model

**3. Model Performance:**

- Perfect 100% accuracy on test data
- No misclassifications
- Model generalizes excellently to unseen data
- Suitable for real-world classification tasks

### Visualizations
<img width="1137" height="986" alt="image" src="https://github.com/user-attachments/assets/8462ddfd-1a92-49a0-a795-51911f729ae8" />
<img width="530" height="455" alt="image" src="https://github.com/user-attachments/assets/36805f33-4f81-4639-999f-59659e60dfce" />


## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- Ronald A. Fisher for creating the original dataset
- scikit-learn community for excellent documentation
- Google Colab for providing free cloud computing resources

