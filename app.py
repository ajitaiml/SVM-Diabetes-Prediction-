import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "diabetes.csv"
diabetes_dataset = pd.read_csv(data_path)

# Splitting data into features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Splitting into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluating different SVM kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_accuracies = {}
for kernel in kernels:
    classifier = svm.SVC(kernel=kernel)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    kernel_accuracies[kernel] = accuracy

# Finding the best kernel
best_kernel = max(kernel_accuracies, key=kernel_accuracies.get)
best_accuracy = kernel_accuracies[best_kernel]

# Training the best model
classifier = svm.SVC(kernel=best_kernel)
classifier.fit(X_train, Y_train)

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Custom CSS for background and center alignment
st.markdown("""
    <style>
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        body {
            background-image: url('https://www.comet.com/site/blog/pima-indian-diabetes-prediction/');
            background-size: cover;
            background-position: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🔍 Diabetes Prediction using SVM")
st.write("### Enter your details to predict the likelihood of diabetes")

# Collecting user input
def user_input_features():
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("🩸 Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("💉 Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("🧑‍⚕️ Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("💊 Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)
    
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    return data

input_data = user_input_features()

# Standardizing input data
input_data_std = scaler.transform(input_data)

# Making predictions
prediction = classifier.predict(input_data_std)

if st.button("🔎 Predict"):
    if prediction[0] == 1:
        st.error("🚨 The person is **Diabetic**")
    else:
        st.success("✅ The person is **Not Diabetic**")

# Displaying model accuracy
st.write(f"### 📈 Best Kernel: **{best_kernel}**")
st.write(f"### 🎯 Model Accuracy: **{best_accuracy:.2%}**")

# Accuracy comparison graph
st.write("### Kernel Comparison")
fig, ax = plt.subplots()
ax.bar(kernel_accuracies.keys(), kernel_accuracies.values(), color=['blue', 'red', 'green', 'purple'])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Kernel Type")
ax.set_title("SVM Kernel Performance")
st.pyplot(fig)

# Data Visualizations
st.write("## 📊 Data Visualizations")

# Distribution of Outcomes
st.write("### Diabetes Outcome Distribution")
fig, ax = plt.subplots()
sns.countplot(x=Y, palette='coolwarm', ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.write("### Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(diabetes_dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

