#dataset from: https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland/data

import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("./dataset/Heart_disease_cleveland_new.csv")

# -------------------------------
# Basic inspection
# -------------------------------
print("Dataset shape:", df.shape)
print("\nDataset info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# Target and feature separation
# -------------------------------
x = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# Data quality checks
# -------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget class distribution:")
print(y.value_counts())
print("\nTarget class distribution (normalized):")
print(y.value_counts(normalize=True))

# -------------------------------
# Save clean reference dataset
# -------------------------------
#df.to_csv("./dataset/heart_clean.csv", index=False)

print("\nClean dataset saved as heart_clean.csv")
