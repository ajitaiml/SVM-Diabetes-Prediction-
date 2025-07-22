# 🩺 Diabetes Prediction using Support Vector Machine (SVM)

This project is a machine learning web application that predicts whether a person has diabetes based on health metrics. It uses a Support Vector Machine (SVM) classifier and achieves an accuracy of **78%**. The app is built using **Streamlit** for interactive UI and **scikit-learn** for model development.

---

## 📌 Project Overview

- **Algorithm Used:** Support Vector Machine (SVM)
- **Accuracy Achieved:** 78%
- **Interface:** Streamlit Web App
- **File to Run:** `Diabetes Prediction.py`

---

## 📂 Features

- Takes user inputs like Glucose, BMI, Age, etc.
- Predicts whether the person is diabetic or not.
- Displays:
  - Prediction result
  - Best-performing SVM kernel
  - Model accuracy
  - Kernel performance graph
  - Outcome distribution in dataset
  - Correlation heatmap of features

---

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features include:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Target: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/diabetes-prediction-svm.git
cd diabetes-prediction-svm
