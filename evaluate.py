import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from preprocess import preprocess

def evaluate_model():
    # Load model & vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Test dataset
    data = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
    X_test = vectorizer.transform([preprocess(doc) for doc in data.data])
    y_test = data.target

    preds = model.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    evaluate_model()

