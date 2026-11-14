
# Breast Cancer Detection System using Machine Learning

## Project Overview
This project develops a machine learning-based Breast Cancer Detection System using the Wisconsin Diagnostic Breast Cancer Dataset.

The system classifies input measurements into:
- Malignant
- Benign

## Technologies Used
- Python
- Google Colab
- VS Code
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

## Project Structure

Breast_Cancer_Detection/
- app/
  - app.py
- models/
  - best_model.pkl
- notebooks/
  - Breast_Cancer_Detection.ipynb
- outputs/
- requirements.txt
- README.md
- .gitignore

## Features
- Data preprocessing
- Exploratory Data Analysis
- Feature correlation analysis
- Model training and comparison
- Hyperparameter tuning
- Model evaluation
- Interactive Streamlit prediction interface

## Installation

Install dependencies:

pip install -r requirements.txt

## Run Application

python -m streamlit run app/app.py

## Dataset
Wisconsin Diagnostic Breast Cancer Dataset (WDBC), available through Scikit-learn.

## Model Evaluation
Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion Matrix

Actual evaluation results are available in the outputs folder.

## Disclaimer
This project is intended for educational purposes only.
It is not a medical diagnostic system.
