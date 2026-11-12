"""Retrain the disease prediction model from data/diseaseData/dataset.csv.
Encoding matches pyscrs/predictDisease.py: symptom -> (index in featureCategories) + 1,
disease -> index in diseaseCategory.
"""
import re, ast, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

src = open("pyscrs/predictDisease.py", encoding="utf8").read()
features = ast.literal_eval(re.search(r"featureCategories = (\[.*?\])\n", src, re.S).group(1))
diseases = ast.literal_eval(re.search(r"diseaseCategory = (\[.*?\])\n", src, re.S).group(1))

df = pd.read_csv("data/diseaseData/dataset.csv")
df = df.fillna("None")
sym_cols = [c for c in df.columns if c != "Disease"]
X = df[sym_cols].apply(lambda col: col.astype(str).str.replace(" ", "", regex=False).map(lambda s: features.index(s) + 1))
y = df["Disease"].map(lambda d: diseases.index(d))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for name, m in [("DecisionTree", DecisionTreeClassifier(random_state=42)),
                ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42))]:
    m.fit(Xtr, ytr)
    print(name, "accuracy:", round(accuracy_score(yte, m.predict(Xte)), 4))

final = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
pickle.dump(final, open("data/diseasePredictModel.pkl", "wb"))
print("saved data/diseasePredictModel.pkl; symptom columns:", len(sym_cols))
