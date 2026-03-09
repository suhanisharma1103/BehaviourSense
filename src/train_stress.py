# src/train_stress.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text_stress
from feature_engineering import get_tfidf_vectorizer
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("../data/raw/stress.csv")  # change filename if needed

# Rename columns if required
df = df.rename(columns={"text": "text", "label": "label"})

# Clean text
df["clean_text"] = df["text"].apply(clean_text_stress)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Vectorization
vectorizer = get_tfidf_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
#model = LogisticRegression(max_iter=1000)
#model.fit(X_train_vec, y_train)
model = LinearSVC()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Stress Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "../models/stress_model.pkl")
joblib.dump(vectorizer, "../models/stress_vectorizer.pkl")

print("Stress model saved successfully.")
