from flask import Flask, request, jsonify
import pickle
from preprocess import preprocess

app = Flask(__name__)

# Load trained model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    # Simple explanation mapping
    explanations = {
        "business": "This text is about Business & Finance.",
        "sport": "This text is about Sports.",
        "politics": "This text is about Politics & Government.",
        "entertainment": "This text is about Entertainment & Media.",
        "tech": "This text is about Technology."
    }

    explanation = explanations.get(prediction, f"This text is about {prediction}.")
    return jsonify({"prediction": prediction, "explanation": explanation})

if __name__ == "__main__":
    app.run(debug=True)
