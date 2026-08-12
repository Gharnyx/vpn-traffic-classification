import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
# Change the path if your file is saved somewhere else
df = pd.read_csv(r"D:\balanced_labeled.csv")

# Keep only numeric columns (remove IP, MAC, etc.)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Label', 'Label_reason']]

X = df[numeric_cols]
y = df['Label']

print(f"Using {X.shape[1]} numeric features")
print(f"Total flows: {X.shape[0]}")
print(f"Label distribution:\n{y.value_counts(normalize=True)}")

# Random Forest with SMOTE pipeline
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# 5-Fold Cross Validation
print("\nRunning 5-Fold Stratified Cross-Validation... (this may take 3-8 minutes)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate scores
acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

# Print Results
print("\n" + "="*65)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*65)
print(f"Accuracy       : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"Macro F1-Score : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"Individual F1 scores: {np.round(f1_scores, 4)}")
print("="*65)

# Save results to Desktop (safer path)
results_df = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': acc_scores,
    'Macro_F1': f1_scores
})
results_df.to_csv(r"C:\Users\hp\Desktop\kfold_results.csv", index=False)

print("\n✅ Results saved to: C:\\Users\\hp\\Desktop\\kfold_results.csv")