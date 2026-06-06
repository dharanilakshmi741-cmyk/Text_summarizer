from flask import Flask, render_template, request
import joblib
from rapidfuzz import process
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ==========================================
# Load Classification Model
# ==========================================

classifier = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

labels = [
    "Technology",
    "Sports",
    "Finance"
]

# ==========================================
# TF-IDF Summarization Function
# ==========================================

def summarize_text(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    # If very small text
    if len(sentences) <= 2:
        return text

    try:

        tfidf = TfidfVectorizer(
            stop_words='english'
        )

        tfidf_matrix = tfidf.fit_transform(
            sentences
        )

        sentence_scores = np.array(
            tfidf_matrix.sum(axis=1)
        ).flatten()

        # Keep around 40-50% sentences
        summary_size = max(
            2,
            int(len(sentences) * 0.5)
        )

        top_indices = sentence_scores.argsort()[
            -summary_size:
        ]

        top_indices = sorted(top_indices)

        summary = " ".join(
            [sentences[i] for i in top_indices]
        )

        return summary

    except Exception as e:

        print("Summarization Error:", e)

        return text


# ==========================================
# Home Route
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    original_text = ""
    summary = ""
    category = ""
    fuzzy_match = ""
    score = ""

    if request.method == "POST":

        original_text = request.form["text"]

        # Summarization
        summary = summarize_text(
            original_text
        )

        # Classification
        text_vector = vectorizer.transform(
            [original_text]
        )

        category = classifier.predict(
            text_vector
        )[0]

        # Fuzzy Matching
        fuzzy_match, score, _ = process.extractOne(
            category,
            labels
        )

    return render_template(
        "index.html",
        original_text=original_text,
        summary=summary,
        category=category,
        fuzzy_match=fuzzy_match,
        score=score
    )


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )