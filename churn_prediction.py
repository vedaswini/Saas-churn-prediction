"""
SaaS / Telco Customer Churn Prediction
---------------------------------------
Predicts whether a customer is likely to churn using a Random Forest
classifier trained on the IBM Telco Customer Churn dataset.

Dataset: data/customer_churn.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
dataset = pd.read_csv("data/customer_churn.csv")

print(dataset.head())
print("Shape:", dataset.shape)
print("Columns:", list(dataset.columns))
print("Missing values per column:\n", dataset.isnull().sum())
print("Churn value counts:\n", dataset["Churn"].value_counts())

# Churn == "Yes": customer left the service
# Churn == "No":  customer stayed

# ---------------------------------------------------------------------------
# 2. Clean up / preprocess
# ---------------------------------------------------------------------------
dataset.drop("customerID", axis=1, inplace=True)

# TotalCharges has a handful of blank entries (new customers with 0 tenure),
# which makes pandas read the column as text. Convert it to numeric first so
# it isn't accidentally label-encoded as a category below.
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(0)

# Encode remaining categorical (text) columns
le = LabelEncoder()
for col in dataset.select_dtypes(include="object").columns:
    dataset[col] = le.fit_transform(dataset[col])

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
x = dataset.drop(columns=["Churn"])
y = dataset["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": x.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature importance:\n", importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(x_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# 7. Predict on a new (example) customer
# ---------------------------------------------------------------------------
# Feature order matches x.columns:
# gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
# MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
# DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
# PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
new_customer = [[1, 0, 1, 0, 12, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 2, 75.5, 850.0]]
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nCustomer is likely to churn")
else:
    print("\nCustomer is likely to stay")
