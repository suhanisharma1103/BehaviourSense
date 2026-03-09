import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import numpy as np
from gemini_service import get_gemini_analysis
#  Stress Model
stress_model_path = "../models/stress_bert_final"
stress_tokenizer = DistilBertTokenizerFast.from_pretrained(stress_model_path)
stress_model = DistilBertForSequenceClassification.from_pretrained(stress_model_path)

#  Toxicity Model
toxic_model_path = "../models/toxic_bert_final"
toxic_tokenizer = DistilBertTokenizerFast.from_pretrained(toxic_model_path)
toxic_model = DistilBertForSequenceClassification.from_pretrained(toxic_model_path)

#  Phishing Model 
phishing_model = joblib.load("../models/phishing_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def get_stress_score(text):
    inputs = stress_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = stress_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs[0][1].item()

def get_toxic_score(text):
    inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = toxic_model(**inputs)
    score = outputs.logits.item()
    return max(0, min(1, score))  

def get_phishing_score(text):
    # Convert text to vector
    vec = vectorizer.transform([text])

    # Get raw SVM score
    score = phishing_model.decision_function(vec)[0]

    # Convert to probability-like score (sigmoid)
    import numpy as np
    prob = 1 / (1 + np.exp(-score))

    return float(prob)


def calculate_risk(text):
    phishing = get_phishing_score(text)
    stress = get_stress_score(text)
    toxicity = get_toxic_score(text)

    final = (0.4 * phishing) + (0.3 * stress) + (0.3 * toxicity)

    gemini_explanation = get_gemini_analysis(text)

    return {
        "phishing_score": round(phishing, 3),
        "stress_score": round(stress, 3),
        "toxicity_score": round(toxicity, 3),
        "final_risk": round(final, 3),
        "gemini_analysis": gemini_explanation
    }

#openapi another risk score- define 0 and 1 - then linear model - then predict again -linear regression
#normal frontend
#then implement extenstion
