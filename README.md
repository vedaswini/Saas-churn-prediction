# SaaS / Telco Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to
churn (cancel their subscription) based on their account and usage data.
Built with a Random Forest classifier on the IBM Telco Customer Churn
dataset.

## Overview

Customer churn is a major cost driver for subscription businesses. This
project trains a classification model to flag customers who are at risk of
leaving, so a business could proactively intervene (offers, support
outreach, etc.) before they churn.

## Dataset

`data/customer_churn.csv` — 7,043 customer records with 21 features,
including:

- Demographics: gender, senior citizen status, partner/dependents
- Account info: tenure, contract type, payment method, paperless billing
- Services: phone, internet, online security/backup, streaming, tech support
- Billing: monthly charges, total charges
- Target: `Churn` (Yes/No)

Source: [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Approach

1. **Preprocessing** — drop the customer ID column, fix `TotalCharges`
   (a few blank entries for brand-new customers are converted to numeric
   and filled with 0), and label-encode categorical columns.
2. **Train/test split** — 80/20 split with a fixed random seed.
3. **Model** — `RandomForestClassifier` from scikit-learn.
4. **Evaluation** — accuracy, classification report, and confusion matrix.
5. **Feature importance** — ranks which features drive the model's
   predictions most, visualized as a horizontal bar chart.

## Results

The model is evaluated on accuracy, precision/recall, and a confusion
matrix (see console output when running the script). Feature importance is
saved as `feature_importance.png`.

## Setup

```bash
git clone https://github.com/vedaswini/saas-churn-prediction.git
cd saas-churn-prediction
pip install -r requirements.txt
python churn_prediction.py
```

## Project structure

```
saas-churn-prediction/
├── churn_prediction.py     # Main training & evaluation script
├── data/
│   └── customer_churn.csv  # Dataset
├── requirements.txt
└── README.md
```

## Tech stack

- Python
- pandas, numpy
- scikit-learn (RandomForestClassifier, LabelEncoder)
- matplotlib

## Future improvements

- Try other models (XGBoost, Logistic Regression) and compare performance
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Handle class imbalance (SMOTE or class weighting)
- Deploy as a simple API or Streamlit app for live predictions
