import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from rapidfuzz import fuzz

# Load Dataset
df = pd.read_csv("data/dataset.csv")

X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(
    max_iter=2000
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nActual Labels:")
print(list(y_test))

print("\nPredicted Labels:")
print(list(predictions))

print(f"\nAccuracy: {accuracy*100:.2f}%")

# Fuzzy Accuracy
total_score = 0

for actual, predicted in zip(y_test, predictions):
    total_score += fuzz.ratio(
        str(actual),
        str(predicted)
    )

fuzzy_accuracy = total_score / len(predictions)

print(f"Fuzzy Accuracy: {fuzzy_accuracy:.2f}%")

# Cross Validation
scores = cross_val_score(
    model,
    X_vec,
    y,
    cv=5
)

print(
    f"Cross Validation Accuracy: {scores.mean()*100:.2f}%"
)

# Save
joblib.dump(
    model,
    "models/classifier.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel Saved Successfully")