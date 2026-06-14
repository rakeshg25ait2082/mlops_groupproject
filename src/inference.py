import os
import json
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# ---- Configuration ----
DEFAULT_MODEL = "Nlp0187/distilbert-imdb-reviews-v1"
DEFAULT_SAMPLE = (
    "This movie was fantastic. The acting was brilliant "
    "and I loved every minute."
)
MAX_LENGTH = 512


def get_device() -> torch.device:
    """Select CUDA if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(repo_id: str, device: torch.device):
    """Load tokenizer and model, move to device, set eval mode."""
    print(f"[inference] Loading model: {repo_id} on {device}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, device: torch.device) -> dict:
    """Run sentiment inference on a single text input."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()

    return {
        "label": model.config.id2label[predicted_id],
        "score": round(confidence, 3),
    }


def main() -> None:
    repo_id = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    device = get_device()
    tokenizer, model = load_model(repo_id, device)
    input_text = os.environ.get("INPUT_TEXT")
    if not input_text:
        input_text = DEFAULT_SAMPLE
        print("[inference] No INPUT_TEXT provided — using default sample.")

    result = predict(input_text, tokenizer, model, device)
    print("\n=== Prediction ===")
    print(f"Input : {input_text.strip()}")
    print(f"Output: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
