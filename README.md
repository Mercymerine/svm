# Support Vector Machine
# Heart Disease Prediction Using Machine Learning
This project aims to predict the presence of heart disease in individuals based on various personal key indicators. The analysis involves data cleaning, encoding, and the application of multiple machine learning models to classify heart disease status.

### Table of Contents
Overview

Dataset

Data Preprocessing

Data Analysis

Modeling

Evaluation

Technologies Used


### Overview
Heart disease is a leading cause of death worldwide. Early detection and accurate prediction can significantly improve treatment outcomes. This project utilizes machine learning algorithms to predict heart disease based on various health indicators.

### Key Goals:
Build a robust and scalable pipeline for data preprocessing, training, and evaluation.
Leverage multiple machine learning algorithms to find the most effective model for prediction.
Visualize data relationships and identify correlations between features.

### Dataset
The dataset used in this project was obtained from Kaggle:

Personal Key Indicators of Heart Disease Dataset

Dataset URL: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

#### Features:
HeartDisease: Target variable indicating the presence of heart disease (Yes/No).
BMI: Body Mass Index.
Smoking: Smoking status (Yes/No).
AlcoholDrinking: Alcohol consumption status (Yes/No).
Stroke: History of stroke (Yes/No).
PhysicalHealth: Number of physically unhealthy days in the past month.
MentalHealth: Number of mentally unhealthy days in the past month.
DiffWalking: Difficulty walking (Yes/No).
Sex: Gender of the individual.
AgeCategory: Age group of the individual.
Race: Race of the individual.
Diabetic: Diabetes status.
PhysicalActivity: Physical activity status (Yes/No).
GenHealth: General health status.
SleepTime: Average sleep time in hours.
Asthma: Asthma status (Yes/No).
KidneyDisease: Kidney disease status (Yes/No).
SkinCancer: Skin cancer status (Yes/No).

### Data Preprocessing
Loading the Data:
The dataset is downloaded from Kaggle and extracted for analysis.

### Data Cleaning:
The dataset is inspected for missing values, and it is found to have no missing data.
Unique values for categorical features are identified.

### Data Encoding:
Label Encoding: The target variable HeartDisease is encoded as binary (Yes=1, No=0).
Ordinal Encoding: Custom encoding is applied to the Diabetic and AgeCategory columns.
One-Hot Encoding: Categorical variables are converted into dummy/indicator variables.

### Data Analysis
Correlation Analysis:
A heatmap is created to visualize the correlation between features, identifying the most correlated factors related to heart disease.

### Modeling
Machine Learning Models Implemented:
Support Vector Machine (SVM):

Model is trained using a polynomial kernel.
Cross-validation is performed to evaluate model accuracy.
K-Nearest Neighbors (KNN):

Model is trained using the k-nearest neighbors algorithm.
Cross-validation is performed to evaluate model accuracy.
Logistic Regression:

A logistic regression model is trained and evaluated using cross-validation.
Naive Bayes:

Gaussian Naive Bayes and Bernoulli Naive Bayes models are trained and evaluated.
Evaluation
Model Performance Metrics:
Test Accuracy: Accuracy of the model on the test dataset.
Classification Report: Precision, recall, and F1-score for each class.
Confusion Matrix: Summary of prediction results.

###### Summary of Results:
SVM Test Accuracy: 83%
KNN Test Accuracy: 81.5%
Logistic Regression Test Accuracy: 84%
Naive Bayes Test Accuracy: 80.8%


### Technologies Used
Programming Language: Python
Libraries:
numpy, pandas for data manipulation
matplotlib, seaborn for visualization
scikit-learn for machine learning
How to Run
