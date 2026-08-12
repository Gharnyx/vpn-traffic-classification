from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
data, meta = arff.loadarff(
    r"C:\Users\hp\Downloads\Scenario A1-ARFF\Scenario A1-ARFF\TimeBasedFeatures-Dataset-15s-VPN.arff"
)

df = pd.DataFrame(data)

# Decode labels
df["class1"] = df["class1"].str.decode("utf-8")

# Separate features and label
X = df.drop("class1", axis=1)
y = df["class1"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importances = model.feature_importances_
features = X.columns

# Sort features
sorted_indices = importances.argsort()

# Plot
plt.figure(figsize=(10,8))
plt.barh(features[sorted_indices], importances[sorted_indices])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance (VPN vs Non-VPN)")
plt.tight_layout()
plt.show()
