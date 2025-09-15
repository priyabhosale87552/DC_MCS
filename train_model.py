import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../dataset/bbc.csv")

# Rename column 'category' → 'label' for compatibility
df.rename(columns={'category': 'label'}, inplace=True)

# Basic check
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must have 'text' and 'label' columns!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a simple classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer if needed
import joblib
joblib.dump(model, 'bbc_model.pkl')
joblib.dump(vectorizer, 'bbc_vectorizer.pkl')
