from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1️⃣ Load ARFF dataset
data, meta = arff.loadarff(
    r"C:\Users\hp\Downloads\Scenario A1-ARFF\Scenario A1-ARFF\TimeBasedFeatures-Dataset-15s-VPN.arff"
)

df = pd.DataFrame(data)

# 2️⃣ Decode byte labels
df["class1"] = df["class1"].str.decode("utf-8")

# 3️⃣ Separate features and label
X = df.drop("class1", axis=1)
y = df["class1"]

# 4️⃣ Train-test split (important: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Predict
y_pred = model.predict(X_test)

# 7️⃣ Print results
print(classification_report(y_test, y_pred))