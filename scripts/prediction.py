import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


#load dataset
df = pd.read_csv('pca_cleaned_data.csv')

#features and target
features = ['track_number', 'artist_popularity', 'artist_followers',
             'track_duration_min', 'PC1','PC2', 'PC3', 'PC4' ]
X = df[features]


POPULARITY_THRESHOLD = 70
y = (df['track_popularity'] >= POPULARITY_THRESHOLD).astype(int)
print(f"Class Distribution:")
print(f"  Unpopular (0): {sum(y == 0)} tracks ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"  Popular (1): {sum(y == 1)} tracks ({sum(y == 1)/len(y)*100:.1f}%)")
print()

# scale_pos_weight = (number of negative examples) / (number of positive examples)
# This tells XGBoost to give more weight to the minority class
scale_pos_weight = sum(y == 0) / sum(y == 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")
print(f"  (XGBoost will weight minority class {scale_pos_weight:.2f}x more)")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify ensures class balance
)

model = XGBClassifier(
    # Basic parameters
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    
    # Class imbalance handling
    scale_pos_weight=scale_pos_weight,  # Weight minority class
    
    # Additional parameters for better performance
    subsample=0.8,              # Use 80% of samples per iteration
    colsample_bytree=0.8,       # Use 80% of features per tree
    min_child_weight=1,         # Minimum weight in child nodes
    gamma=0,                    # Minimum loss reduction for split
    
    # Evaluation metric for imbalanced data
    eval_metric='logloss',      # Better for imbalanced classification    
    verbosity=1
)


# Train model
print("Training XGBoost Classifier...")
model.fit(X_train, y_train)
print("✓ Model training complete!\n")


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

print("=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

# Standard metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (of predicted popular, how many are actually popular)")
print(f"Recall:    {recall:.4f}    (of actual popular, how many did we catch)")
print(f"F1 Score:  {f1:.4f}     (harmonic mean of precision & recall)")
print(f"ROC-AUC:   {roc_auc:.4f}     (probability model ranks popular higher)\n")


# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Unpopular', 'Popular']))



#cross validation
print("=" * 70)
print("CROSS-VALIDATION COMPARISON")
print("=" * 70)

# 1. Stratified K-Fold (Recommended for imbalanced data)
print("\n1. Stratified K-Fold CV (Best for imbalanced data):")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_skfold = cross_val_score(model, X, y, cv=skfold, scoring='f1')
print(f"   F1 Scores: {cv_scores_skfold}")
print(f"   Mean F1: {cv_scores_skfold.mean():.4f} (+/- {cv_scores_skfold.std():.4f})")

# 2. Stratified Group K-Fold (By artist)
print("\n2. Stratified Group K-Fold CV (By artist - prevents leakage):")
try:
    artist_groups = pd.factorize(df['artist_name'])[0]
    sgkfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_sgkfold = cross_val_score(
        model, X, y, groups=artist_groups, cv=sgkfold, scoring='f1'
    )
    print(f"   F1 Scores: {cv_scores_sgkfold}")
    print(f"   Mean F1: {cv_scores_sgkfold.mean():.4f} (+/- {cv_scores_sgkfold.std():.4f})")
except Exception as e:
    print(f"   Could not perform GroupKFold: {e}")



feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))




import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and grid
# 6. Cross-Validation Comparison
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)


ax6 = fig.add_subplot(gs[2, :2])
cv_methods = ['Stratified K-Fold', 'Stratified Group K-Fold']
cv_means = [cv_scores_skfold.mean(), cv_scores_sgkfold.mean()]
cv_stds = [cv_scores_skfold.std(), cv_scores_sgkfold.std()]
x_pos = np.arange(len(cv_methods))
ax6.bar(x_pos, cv_means, yerr=cv_stds, capsize=10, alpha=0.7, 
        color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.5)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(cv_methods)
ax6.set_ylabel('F1 Score')
ax6.set_title('Cross-Validation Method Comparison')
ax6.set_ylim([0.85, 1.0])
ax6.grid(axis='y', alpha=0.3)
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax6.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', fontweight='bold')



import joblib   
# SAVE the model
joblib.dump(model, 'spotify_xgboost_model.pkl')
print("Model saved successfully!")


# -saving all the metrices to a text file

# Define the filename
report_filename = "model_performance_metrics.txt"

with open(report_filename, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("SPOTIFY POPULARITY PREDICTION: MODEL REPORT\n")
    f.write("=" * 70 + "\n\n")

    # 1. Basic Metrics
    f.write("OVERALL PERFORMANCE:\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")

    # 2. Detailed Report
    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Unpopular', 'Popular']))
    f.write("\n")

    # 3. Cross-Validation Results
    f.write("=" * 70 + "\n")
    f.write("CROSS-VALIDATION RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Stratified K-Fold Mean F1:       {cv_scores_skfold.mean():.4f}\n")
    try:
        f.write(f"Stratified Group K-Fold Mean F1: {cv_scores_sgkfold.mean():.4f}\n")
    except NameError:
        f.write("Stratified Group K-Fold: Not available\n")
    f.write("\n")

    # 4. Feature Importance
    f.write("FEATURE IMPORTANCE RANKING:\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\n")
    f.write("Report generated successfully.")

print(f"✓ Metrics saved to {report_filename}")