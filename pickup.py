import pandas as pd

df = pd.read_csv("D:/balanced_labeled.csv")

sample_df = df.sample(n=150, random_state=42)
sample_df.to_csv("validation_sample.csv", index=False)