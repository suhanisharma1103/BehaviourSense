import pandas as pd
import nltk
import re
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)




df = pd.read_csv(
    "../data/raw/spam.csv",
    header=None,
    encoding="latin-1"
)

df[['label', 'text']] = df[0].str.split('\t', n=1, expand=True)

df = df[['label', 'text']]

# Clean labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Remove broken rows
df = df.dropna()

print("Dataset loaded successfully!")
print(df["label"].value_counts())

df['text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#model = LogisticRegression(class_weight='balanced')
from sklearn.svm import LinearSVC
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("../models/phishing_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")
