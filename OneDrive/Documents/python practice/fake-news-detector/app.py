import os
if not os.path.exists("model/model.pkl"):
    import subprocess
    subprocess.run(["python", "train_model.py"])
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load all models
MODEL_PATH      = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")
SCALER_PATH     = os.path.join("model", "scaler.pkl")
LR_PATH         = os.path.join("model", "lr.pkl")
NB_PATH         = os.path.join("model", "nb.pkl")
RF_PATH         = os.path.join("model", "rf.pkl")
ACC_PATH        = os.path.join("model", "accuracies.pkl")

with open(MODEL_PATH,      "rb") as f: model      = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
with open(SCALER_PATH,     "rb") as f: scaler     = pickle.load(f)
with open(LR_PATH,         "rb") as f: lr         = pickle.load(f)
with open(NB_PATH,         "rb") as f: nb         = pickle.load(f)
with open(RF_PATH,         "rb") as f: rf         = pickle.load(f)
with open(ACC_PATH,        "rb") as f: accuracies = pickle.load(f)

print("All models loaded successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/accuracies", methods=["GET"])
def get_accuracies():
    return jsonify(accuracies)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "No text provided."}), 400

    text = data["text"].strip()

    if len(text) < 20:
        return jsonify({"error": "Please enter at least 20 characters."}), 400

    # Vectorize
    text_tfidf  = vectorizer.transform([text])
    text_scaled = scaler.transform(text_tfidf)

    # All model predictions
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # PAC
    pac_pred  = model.predict(text_tfidf)[0]
    pac_score = sigmoid(abs(model.decision_function(text_tfidf)[0])) * 100
    pac_conf  = float(np.clip(pac_score, 50.0, 99.9))

    # Logistic Regression
    lr_pred   = lr.predict(text_scaled)[0]
    lr_proba  = lr.predict_proba(text_scaled)[0]
    lr_conf   = float(np.clip(max(lr_proba) * 100, 50.0, 99.9))

    # Naive Bayes
    nb_pred   = nb.predict(text_tfidf)[0]
    nb_proba  = nb.predict_proba(text_tfidf)[0]
    nb_conf   = float(np.clip(max(nb_proba) * 100, 50.0, 99.9))

    # Random Forest
    rf_pred   = rf.predict(text_scaled)[0]
    rf_proba  = rf.predict_proba(text_scaled)[0]
    rf_conf   = float(np.clip(max(rf_proba) * 100, 50.0, 99.9))

    # Final prediction by majority vote
    votes     = [pac_pred, lr_pred, nb_pred, rf_pred]
    final     = "REAL" if votes.count("REAL") >= 2 else "FAKE"
    avg_conf  = round((pac_conf + lr_conf + nb_conf + rf_conf) / 4, 1)

    return jsonify({
        "prediction"  : final,
        "confidence"  : avg_conf,
        "label_class" : final.lower(),
        "models": {
            "PAC"                : {"pred": pac_pred, "conf": round(pac_conf, 1)},
            "Logistic Regression": {"pred": lr_pred,  "conf": round(lr_conf,  1)},
            "Naive Bayes"        : {"pred": nb_pred,  "conf": round(nb_conf,  1)},
            "Random Forest"      : {"pred": rf_pred,  "conf": round(rf_conf,  1)},
        }
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)