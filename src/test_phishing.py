import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Load model
model = pickle.load(open("../models/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

print("Phishing Detection System Ready!")
print("Type 'exit' to quit\n")

while True:
    message = input("Enter message: ")

    if message.lower() == "exit":
        break

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        print("⚠️  SPAM DETECTED\n")
    else:
        print("✅ SAFE MESSAGE\n")
