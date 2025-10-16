# gene_expr_project.py
# Author: Sameera Ilyas
# Project: Gene Expression Cancer Classification (Bioinformatics + ML)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("🔬 Loading data...")
data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

print("✅ Data loaded successfully!")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# The labels might be in a column named 'Class' or 'CancerType'
# Adjust this line if needed:
# Use only one label column (if there are two)
if labels.shape[1] > 1:
    print("⚠️ Multiple columns found in labels.csv. Using the second one for y.")
    y = labels.iloc[:, 1]
else:
    y = labels.iloc[:, 0]

X = data.select_dtypes(include=["number"])
print("Columns used for training:", X.columns[:10])



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Unique samples in total:", len(data))
print("Unique samples in train set:", len(X_train))
print("Unique samples in test set:", len(X_test))


print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

model.fit(X_train, y_train)

print("🔎 Evaluating...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance visualization
importances = model.feature_importances_
indices = importances.argsort()[::-1][:15]  # top 15 genes

plt.figure(figsize=(10,6))
plt.title("Top 15 Gene Feature Importances")
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [data.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

print("\n✅ Project completed successfully!")
