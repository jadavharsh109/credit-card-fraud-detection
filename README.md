# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Domain](https://img.shields.io/badge/Domain-FinTech%20%2F%20Fraud%20Detection-red)

An end-to-end Machine Learning project to identify fraudulent credit card transactions using classification algorithms, addressing extreme class imbalance.

## 📌 Project Overview
Credit card fraud causes billions of dollars in losses annually. Detecting fraudulent transactions in real-time requires high precision and recall while minimizing false positives for legitimate customers. This project analyzes anonymized credit card transaction data and trains classification models to accurately distinguish between fraudulent and genuine transactions.

## 📁 Project Structure
```text
credit-card-fraud-detection/
├── data/
│   ├── .gitkeep
│   └── README.md                                  # Instructions for obtaining the Kaggle dataset
├── notebooks/
│   └── credit_card_fraud_detection.ipynb          # EDA, preprocessing, model training & evaluation
└── README.md
```

## 📊 Dataset Details
The project utilizes the widely-recognized Kaggle Credit Card Fraud Detection dataset:
- **Total Transactions:** 284,807 transactions
- **Features:** 28 PCA-transformed numerical features (`V1` to `V28`), `Time`, and `Amount`
- **Target (`Class`):** `0` = Genuine transaction, `1` = Fraudulent transaction
- **Class Imbalance:** Only ~0.172% of transactions are fraudulent (highly imbalanced dataset)

> **Note:** To obtain the dataset, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it into the `data/` folder.

## 🛠️ Tech Stack & Workflow
- **Language:** Python
- **Core Libraries:**
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn` for exploratory data analysis
  - `scikit-learn` for preprocessing, modeling, and evaluation metrics
- **Workflow:**
  1. **Exploratory Data Analysis (EDA):** Distribution analysis of transaction amounts, time, and class imbalance.
  2. **Data Preprocessing:** Standard scaling of `Amount` and `Time` features.
  3. **Modeling:** Logistic Regression and Random Forest Classifier.
  4. **Evaluation:** Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curve.

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jadavharsh109/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. **Place dataset:**
   Download `creditcard.csv` and place it inside `data/`.
4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/credit_card_fraud_detection.ipynb
   ```
