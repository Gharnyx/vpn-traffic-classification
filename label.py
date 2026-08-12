from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff(
    r"C:\Users\hp\Downloads\Scenario A1-ARFF\Scenario A1-ARFF\TimeBasedFeatures-Dataset-15s-VPN.arff"
)

df = pd.DataFrame(data)

# Decode byte labels
df["class1"] = df["class1"].str.decode("utf-8")

print(df["class1"].value_counts())