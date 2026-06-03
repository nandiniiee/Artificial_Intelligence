#importing libraries and loading the dataset
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("diabetes.csv")
print("data in the csv file\n")
print(df.head())
print("\n")

#inspecting data struture and missing values
df.info()
print("number of missing values: ")
print(df.isnull().sum())
print("\n")

#statistical summary of the dataset
print("statistical summary of the dataset\n")
df.describe()
fig, axs = plt.subplots(len(df.columns), 1, figsize=(7, 18), dpi=95)
for i, col in enumerate(df.columns):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
plt.tight_layout()
plt.show()

#removing outliers using IQR method
q1, q3 = np.percentile(df['Insulin'], [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
clean_df = df[(df['Insulin'] >= lower) & (df['Insulin'] <= upper)]

#correlation heatmap
corr = df.corr()
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
print(corr['Outcome'].sort_values(ascending=False))

# checking if dibates v/s non-diabetes are balanced
plt.pie(clean_df['Outcome'].value_counts(),
        labels=['Diabetes', 'Not Diabetes'],
        autopct='%.f%%', shadow=True)
plt.title('Outcome Proportionality')       
plt.show()

# preparing separate vraibles and feature targets
X = df.drop(columns=['Outcome'])
y = df['Outcome']

#feature scaling using standardization

#1. Nonrmal min-max scaling
scaler=MinMaxScaler()
X_normalized=scaler.fit_transform(X)
print(X_normalized[:5])

#2. Standardization
scaler=StandardScaler()
X_standardized=scaler.fit_transform(X)
print(X_standardized[:5])
