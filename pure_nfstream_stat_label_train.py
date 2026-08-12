# unbiased_vpn_train.py
# Adapted from ODE-Flow GitHub — trains RF with SMOTE on NFStream features
# Handles imbalance for unbiased VPN/non-VPN classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load your data
df = pd.read_csv(r"D:\traffic_nfstream.csv")

# Select key features (IAT, packet size, duration, symmetry — like ODE-Flow)
features = [
    'bidirectional_stddev_piat_ms', 'bidirectional_mean_piat_ms',  # IAT std/mean
    'bidirectional_stddev_ps', 'bidirectional_mean_ps',           # Packet size std/mean
    'bidirectional_duration_ms',                                 # Duration
    'src2dst_packets', 'dst2src_packets',                        # Symmetry
    'bidirectional_packets', 'bidirectional_bytes'                # Counts
]
X = df[features].fillna(0)

# Add auto-labels (from previous rules — replace with your labeled column if you have it)
# For demo, use a simple rule-based label (low IAT stddev + symmetry = VPN)
ratio = df['src2dst_packets'] / (df['dst2src_packets'] + 1e-6)
y = ((X['bidirectional_stddev_piat_ms'] < 30) & ratio.between(0.8, 1.2)).astype(int)  # 0/1 label

print("Label distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# SMOTE for unbiased training (oversample minority)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train RF
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)

y_pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-VPN (0)', 'VPN (1)']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (like ODE-Flow)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop Features:")
print(importances)

# Save
pd.DataFrame({'True': y_test, 'Predicted': y_pred}).to_csv(r"D:\unbiased_predictions.csv", index=False)
print("\nPredictions saved to D:\\unbiased_predictions.csv")