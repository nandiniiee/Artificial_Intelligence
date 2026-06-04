#imporitng libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, decomposition
data=pd.read_csv("Employee_dataset.csv")
print("data in the csv file\n")
print(data.head())
print("\n")
#-------UNIVARIATE ANALYSIS----------------
#using histogram
plt.figure(figsize=(10, 5))
sns.histplot(data['age'])
plt.title("Age distribution")
plt.savefig('employee_histogram.png', dpi=150, bbox_inches='tight')

#bar chart
plt.figure(figsize=(10, 5))
sns.countplot(data['gender_full'])
plt.title("Gender distribution")
plt.savefig('employee_barChart.png', dpi=150, bbox_inches='tight')

#pie chart
plt.figure()
x=data['STATUS_YEAR'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.savefig('employee_pieChart.png', dpi=150, bbox_inches='tight')

#displaying all the plots
plt.show()

#-------BIVARIATE ANALYSIS----------------

#1. categorical v/s Numerical
plt.figure(figsize=(15, 5))
sns.barplot(x=data['department_name'],
            y=data['length_of_service'])
plt.xticks(rotation=90)
plt.title("Department vs Length of Service")
plt.savefig('employee_categorical_vs_numerical.png', dpi=150, bbox_inches='tight')

#Numerical vs Numerical
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data['length_of_service'], y=data['age'])
plt.title("Length of Service vs Age")
plt.savefig('employee_numerical_vs_numerical.png', dpi=150, bbox_inches='tight')

#categorical vs categorical
plt.figure(figsize=(10, 5))
sns.countplot(x='STATUS_YEAR', hue='STATUS', data=data)
plt.title("Status Year vs Status")
plt.savefig('employee_statusyear_vs_status.png', dpi=150, bbox_inches='tight')

#displaying all the plots
plt.show()

#----MULTIVARIATE ANALYSIS----------------
#1.PCA
plt.figure()
iris = datasets.load_iris()
X=iris.data
y=iris.target
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(X)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.title("PCA of Iris Dataset")
plt.savefig('employee_pca.png', dpi=150, bbox_inches='tight')

#2.HeatMap
plt.figure()
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True)
plt.title("Correlation Heatmap ")
plt.savefig('employee_heatmap.png', dpi=150, bbox_inches='tight')

#displaying all the plots
plt.show()
