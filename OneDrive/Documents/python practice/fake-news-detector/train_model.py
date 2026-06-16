import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler

# ── Load Kaggle Dataset ──
print("Loading Kaggle dataset...")

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine title + text
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]
true_df["content"] = true_df["title"] + " " + true_df["text"]

# Use 5000 samples each for speed
fake_sample = fake_df[["content","label"]].sample(5000, random_state=42)
true_sample = true_df[["content","label"]].sample(5000, random_state=42)

df = pd.concat([fake_sample, true_sample])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset ready: {len(df)} articles")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ── TF-IDF ──
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled  = scaler.transform(X_test_tfidf)

# ── Train Models ──
print("\nTraining models...")

pac = PassiveAggressiveClassifier(max_iter=50, random_state=42, C=0.5)
pac.fit(X_train_tfidf, y_train)
pac_acc = accuracy_score(y_test, pac.predict(X_test_tfidf))
print(f"PAC Accuracy: {pac_acc*100:.2f}%")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test_tfidf))
print(f"Naive Bayes Accuracy: {nb_acc*100:.2f}%")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# ── Save ──
os.makedirs("model", exist_ok=True)

accuracies = {
    "Passive Aggressive" : round(pac_acc*100, 2),
    "Logistic Regression": round(lr_acc*100,  2),
    "Naive Bayes"        : round(nb_acc*100,  2),
    "Random Forest"      : round(rf_acc*100,  2)
}

with open("model/model.pkl",      "wb") as f: pickle.dump(pac,        f)
with open("model/vectorizer.pkl", "wb") as f: pickle.dump(tfidf,      f)
with open("model/scaler.pkl",     "wb") as f: pickle.dump(scaler,     f)
with open("model/lr.pkl",         "wb") as f: pickle.dump(lr,         f)
with open("model/nb.pkl",         "wb") as f: pickle.dump(nb,         f)
with open("model/rf.pkl",         "wb") as f: pickle.dump(rf,         f)
with open("model/accuracies.pkl", "wb") as f: pickle.dump(accuracies, f)

print("\nAll models saved!")
print("Now run: python app.py")