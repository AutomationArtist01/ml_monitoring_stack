import pandas as pd

# Dataset path
file_path = "data/ML DATASET_Telco-Customer-Churn.csv"

# Load dataset
df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)
print(df["Churn"].value_counts())

print("\n" + "=" * 60)
print("CHURN PERCENTAGE")
print("=" * 60)
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 60)

for column in df.select_dtypes(include="object").columns:
    print(f"\n{column}:")
    print(df[column].unique())

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())
