import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("../data/raw/toxic.csv")

print("Columns in dataset:", df.columns)


df = df.rename(columns={
    'body': 'comment',
    'score': 'offensiveness_score'
})

# Keep only required columns
df = df[['comment', 'offensiveness_score']]
df = df.dropna()

print("Dataset size after cleaning:", len(df))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment'].tolist(),
    df['offensiveness_score'].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)



class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset = ToxicDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1   
)


training_args = TrainingArguments(
    output_dir="../models/toxic_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./logs",
    load_best_model_at_end=True
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.flatten()
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)



trainer.train()



trainer.save_model("../models/toxic_bert_final")
tokenizer.save_pretrained("../models/toxic_bert_final")

print("✅ Toxicity model saved successfully!")
