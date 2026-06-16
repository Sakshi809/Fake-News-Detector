import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH      = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Model files not found! Please run 'python train_model.py' first."
    )

with open(MODEL_PATH,      "rb") as f: model      = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "No text provided."}), 400

    text = data["text"].strip()

    if len(text) < 20:
        return jsonify({"error": "Please enter at least 20 characters."}), 400

    text_tfidf   = vectorizer.transform([text])
    prediction   = model.predict(text_tfidf)[0]

    raw_score    = model.decision_function(text_tfidf)[0]
    sigmoid      = lambda x: 1 / (1 + np.exp(-x))
    confidence   = sigmoid(abs(raw_score)) * 100
    confidence   = float(np.clip(confidence, 50.0, 99.9))

    return jsonify({
        "prediction"  : prediction,
        "confidence"  : round(confidence, 1),
        "label_class" : prediction.lower()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)