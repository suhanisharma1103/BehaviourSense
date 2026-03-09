# src/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------
# Generic TF-IDF (Phishing)
# -------------------------
def get_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )


# -------------------------
# Stress-specific TF-IDF
# -------------------------
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        min_df=2
    )
