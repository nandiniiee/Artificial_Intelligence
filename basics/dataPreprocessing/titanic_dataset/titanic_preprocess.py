import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("Titanic-Dataset.csv")

print("--- DataFrame Info ---")
df.info()  # This prints automatically as it is a built-in method

print("\n--- DataFrame Head ---")
print(df.head())  # Added print()

print("\n--- Duplicated Rows Count ---")
print(df.duplicated().sum())  # Added .sum() and print() to show total duplicates

# 2. Separate Columns
cat_col = [col for col in df.columns if df[col].dtype == "object"]
num_col = [col for col in df.columns if df[col].dtype != "object"]

print("\nCategorical columns:", cat_col)
print("Numerical columns:", num_col)

print("\n--- Unique Values in Categorical Columns ---")
print(df[cat_col].nunique())  # Added print()

print("\n--- Missing Values Percentage ---")
print(round((df.isnull().sum() / df.shape[0]) * 100, 2))  # Added print()

# 3. Clean Data
df1 = df.drop(columns=["Name", "Ticket", "Cabin"])
df1.dropna(subset=["Embarked"], inplace=True)
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())

# 4. Outlier Filter 1 (2 Standard Deviations)
mean = df1["Age"].mean()
std = df1["Age"].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

df2 = df1[(df1["Age"] >= lower_bound) & (df1["Age"] <= upper_bound)]

# 5. Outlier Filter 2
df3 = df2.fillna(df2["Age"].mean())

print("\n--- Missing Values in df3 ---")
print(df3.isnull().sum())  # Added print()

mean = df3["Age"].mean()
std = df3["Age"].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print("\nLower Bound :", lower_bound)
print("Upper Bound :", upper_bound)

df4 = df3[(df3["Age"] >= lower_bound) & (df3["Age"] <= upper_bound)]
print(f"\n✅ Final dataset shape after outlier removal: {df4.shape}")

# 6. Plotting (Placed at the end so it doesn't freeze your calculations)
plt.boxplot(df1["Age"], vert=False)
plt.ylabel("Variable")
plt.xlabel("Age")
plt.title("Box Plot")
print("\n📊 Displaying Box Plot... (Close the plot window to finish the script)")
plt.show()
