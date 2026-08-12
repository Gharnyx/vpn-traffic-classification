from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (VPN vs Non-VPN)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0,1], ["Non-VPN", "VPN"])
plt.yticks([0,1], ["Non-VPN", "VPN"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center")

plt.show()