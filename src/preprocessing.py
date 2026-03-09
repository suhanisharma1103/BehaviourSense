# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# -------------------------
# Phishing Preprocessing
# -------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# -------------------------
# Stress Preprocessing
# -------------------------
def clean_text_stress(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Reddit mentions
    text = re.sub(r"u\/\w+", "", text)
    text = re.sub(r"r\/\w+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)
