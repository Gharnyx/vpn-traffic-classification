import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv(r"D:\balanced_labeled.csv")   # Change path if needed

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Label', 'Label_reason']]

X = df[numeric_cols]
y = df['Label']

print(f"Using {X.shape[1]} numeric features")
print(f"Total flows: {X.shape[0]}")
print(f"Label distribution:\n{y.value_counts(normalize=True)}")

# 5-Fold CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', random_state=42, probability=True),
    "XGBoost": XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1, eval_metric='logloss')
}

print("\n" + "="*70)
print("BASELINE COMPARISON - 5-FOLD CROSS-VALIDATION")
print("="*70)

results = []

for name, model in models.items():
    print(f"\nRunning {name}...")
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    print(f"Accuracy  : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Macro F1  : {f1.mean():.4f} ± {f1.std():.4f}")
    
    results.append({
        "Model": name,
        "Accuracy_Mean": acc.mean(),
        "Accuracy_Std": acc.std(),
        "MacroF1_Mean": f1.mean(),
        "MacroF1_Std": f1.std()
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\Users\hp\Desktop\baselines_comparison.csv", index=False)
print("\n✅ Baseline results saved to: C:\\Users\\hp\\Desktop\\baselines_comparison.csv")