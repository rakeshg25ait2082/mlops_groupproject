"""Run sentiment inference using a fine-tuned DistilBERT model."""

import os
import json
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# Read model name from env (set by Docker ARG/ENV); fallback to default
HF_REPO_ID = os.environ.get(
    "HF_MODEL_NAME",
    "Nlp0187/distilbert-imdb-reviews-v1"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[inference] Loading model: {HF_REPO_ID} on {DEVICE}")

tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO_ID)
model = DistilBertForSequenceClassification.from_pretrained(HF_REPO_ID)
model.to(DEVICE)
model.eval()


def predict(text: str):
    """Predict sentiment label and confidence score for an input string."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()

    return {
        "label": model.config.id2label[predicted_id],
        "score": round(confidence, 4),
    }


if __name__ == "__main__":
    input_text = os.environ.get("INPUT_TEXT")

    if not input_text:
        input_text = (
            "This movie was fantastic. The acting was brilliant "
            "and I loved every minute."
        )
        print("[inference] No INPUT_TEXT provided — using default sample.")

    result = predict(input_text)
    print("\n=== Prediction ===")
    print(f"Input : {input_text.strip()}")
    print(f"Output: {json.dumps(result, indent=2)}")
