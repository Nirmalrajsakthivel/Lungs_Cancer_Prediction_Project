# 🫁 Lung Cancer Prediction using Machine Learning

This project uses machine learning to predict the likelihood of lung cancer based on various health and lifestyle attributes such as smoking, anxiety, peer pressure, and more.

## 📌 Project Overview

Lung cancer is one of the leading causes of death globally. Early detection can significantly improve treatment outcomes. This project applies supervised machine learning to classify whether a person is likely to have lung cancer based on the input data.

---

## 📂 Files in the Repository

📊 Dataset
The dataset consists of 309 patient records and 16 features including symptoms and health habits.

🔽 How to Upload the Dataset
The dataset is required for training the model. Use one of the following options:

✅ Option 1: Dataset Included
If you have cloned the repository, the dataset is already present in the data/ folder:

Path: data/lung_cancer_dataset.csv

Make sure the dataset path in the script matches this:

df = pd.read_csv("data/lung_cancer_dataset.csv")

✅ Option 2: Manual Upload
If the dataset is not present, you can download it here and place it inside a data/ directory:

🧠 Features Used
GENDER
AGE
SMOKING
YELLOW_FINGERS
ANXIETY
PEER_PRESSURE
CHRONIC DISEASE
FATIGUE
ALLERGY
WHEEZING
ALCOHOL CONSUMING
COUGHING
SHORTNESS OF BREATH
SWALLOWING DIFFICULTY
CHEST PAIN
Target Variable: LUNG_CANCER (YES/NO)

📈 Model Used
Random Forest Classifier
Accuracy: ~99.6%
Precision & Recall: ~98.3%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv("data/lung_cancer_dataset.csv")

# Convert categorical values
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Features and label
X = df.drop("LUNG_CANCER", axis=1)

y = df["LUNG_CANCER"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Chart 

![Screenshot 2025-04-01 224220](https://github.com/user-attachments/assets/04ff4e1b-4386-49b0-8ecb-5250ffe7db9f)

![Screenshot 2025-04-01 224317](https://github.com/user-attachments/assets/1bff615f-9ea5-4e61-ae65-6bdc5bc920bf)

![Screenshot 2025-04-01 225032](https://github.com/user-attachments/assets/229a8a2a-eff1-410e-bd9e-fa486c9ef63d)

model = RandomForestClassifier()

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

# Prediction example
sample = [[1, 65, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2]]

prediction = model.predict(sample)

print("Prediction:", prediction)

The person does not have lung cancer.
