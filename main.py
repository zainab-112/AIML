import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target

print("First 5 Rows of Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(5,8))
sns.countplot(x='target', data=df)
plt.title("Class Distribution")
plt.xticks([0,1], ['Malignant', 'Benign'])
plt.show()

df.hist(figsize=(15,12))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
