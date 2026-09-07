import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import joblib

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'creditcard.csv')
assets_dir = os.path.join(base_dir, 'assets')
models_dir = os.path.join(base_dir, 'models')

os.makedirs(assets_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("1. Loading dataset...")
df = pd.read_csv(data_path)
total_tx = len(df)
fraud_tx = int(df['Class'].sum())
genuine_tx = total_tx - fraud_tx
print(f"Total Transactions: {total_tx:,}")
print(f"Genuine: {genuine_tx:,} ({genuine_tx/total_tx*100:.2f}%)")
print(f"Fraud:   {fraud_tx:,} ({fraud_tx/total_tx*100:.3f}%)")

# 2. Stratified Train-Test Split (Zero Data Leakage)
print("\n2. Performing Stratified Train-Test Split (80/20)...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training shape: {X_train.shape} (Frauds: {y_train.sum()})")
print(f"Testing shape:  {X_test.shape} (Frauds: {y_test.sum()})")

# 3. Scaling strictly fitted on train set ONLY
print("\n3. Scaling 'Time' and 'Amount' (Fitted strictly on X_train)...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
print("Scaler saved to models/scaler.joblib")

# 4. Train Models
print("\n4. Training Baseline Logistic Regression (class_weight='balanced')...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n5. Training Random Forest Classifier (class_weight='balanced', n_estimators=100)...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=14, class_weight='balanced',
    n_jobs=-1, random_state=42
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

joblib.dump(rf, os.path.join(models_dir, 'random_forest_fraud_model.joblib'))
print("Random Forest model saved to models/random_forest_fraud_model.joblib")

# 6. Evaluation Metrics
rep_lr = classification_report(y_test, y_pred_lr, output_dict=True)
rep_rf = classification_report(y_test, y_pred_rf, output_dict=True)

auprc_lr = average_precision_score(y_test, y_prob_lr)
auprc_rf = average_precision_score(y_test, y_prob_rf)

roc_lr = roc_auc_score(y_test, y_prob_lr)
roc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n=======================================================")
print("             MODEL EVALUATION REPORT (CLASS 1 - FRAUD)   ")
print("=======================================================")
print(f"{'Metric':<25} | {'Logistic Regression':<20} | {'Random Forest':<20}")
print("-" * 71)
print(f"{'Fraud Precision':<25} | {rep_lr['1']['precision']*100:>18.2f}% | {rep_rf['1']['precision']*100:>18.2f}%")
print(f"{'Fraud Recall':<25} | {rep_lr['1']['recall']*100:>18.2f}% | {rep_rf['1']['recall']*100:>18.2f}%")
print(f"{'Fraud F1-Score':<25} | {rep_lr['1']['f1-score']*100:>18.2f}% | {rep_rf['1']['f1-score']*100:>18.2f}%")
print(f"{'AUPRC (Avg Precision)':<25} | {auprc_lr*100:>18.2f}% | {auprc_rf*100:>18.2f}%")
print(f"{'ROC-AUC':<25} | {roc_lr*100:>18.2f}% | {roc_rf*100:>18.2f}%")
print("=======================================================\n")

# 7. Generate & Save Visualizations
sns.set_theme(style='whitegrid')

# Visual A: Side-by-side Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_lr, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'], annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title(f"Logistic Regression (Balanced)\nPrecision: {rep_lr['1']['precision']*100:.1f}% | Recall: {rep_lr['1']['recall']*100:.1f}%", fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual Class', fontweight='bold')
axes[0].set_xlabel('Predicted Class', fontweight='bold')

sns.heatmap(cm_rf, annot=True, fmt=',d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'], annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title(f"Random Forest (Balanced)\nPrecision: {rep_rf['1']['precision']*100:.1f}% | Recall: {rep_rf['1']['recall']*100:.1f}%", fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual Class', fontweight='bold')
axes[1].set_xlabel('Predicted Class', fontweight='bold')

plt.tight_layout()
cm_path = os.path.join(assets_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Saved: {cm_path}")

# Visual B: ROC Curves Comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='#3b82f6', lw=2.5, label=f'Logistic Regression (AUC = {roc_lr:.4f})')
plt.plot(fpr_rf, tpr_rf, color='#10b981', lw=2.5, label=f'Random Forest (AUC = {roc_rf:.4f})')
plt.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=11)
plt.tight_layout()
roc_path = os.path.join(assets_dir, 'roc_curve.png')
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"Saved: {roc_path}")

# Visual C: Precision-Recall Curve Comparison (Key for Imbalance!)
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(rec_lr, prec_lr, color='#3b82f6', lw=2.5, label=f'Logistic Regression (AUPRC = {auprc_lr:.4f})')
plt.plot(rec_rf, prec_rf, color='#10b981', lw=2.5, label=f'Random Forest (AUPRC = {auprc_rf:.4f})')
no_skill = fraud_tx / total_tx
plt.plot([0, 1], [no_skill, no_skill], color='#ef4444', lw=1.5, linestyle='--', label=f'Baseline Baseline ({no_skill*100:.3f}%)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Fraud Detection Rate)', fontweight='bold')
plt.ylabel('Precision (True Positive Accuracy)', fontweight='bold')
plt.title('Precision-Recall Curve (Imbalanced Benchmark)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', frameon=True, fontsize=11)
plt.tight_layout()
pr_path = os.path.join(assets_dir, 'precision_recall_curve.png')
plt.savefig(pr_path, dpi=200)
plt.close()
print(f"Saved: {pr_path}")

# Visual D: Top 10 Feature Importances
importances = rf.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(9, 5))
sns.barplot(x=importances[indices], y=[feat_names[i] for i in indices], palette='viridis')
plt.title('Top 10 Most Predictive Features for Fraud Detection (Random Forest)', fontsize=13, fontweight='bold')
plt.xlabel('Gini Feature Importance Score', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')
plt.tight_layout()
fi_path = os.path.join(assets_dir, 'feature_importance.png')
plt.savefig(fi_path, dpi=200)
plt.close()
print(f"Saved: {fi_path}")

print("\nPipeline execution complete!")
