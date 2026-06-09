# MLOps Group Project — Sentiment Analysis with DistilBERT

Fine-tuned **DistilBERT** on the IMDB movie reviews dataset for binary sentiment classification (Positive / Negative). The project covers the full MLOps lifecycle: training with experiment tracking, model registry on Hugging Face, containerised inference via Docker, and versioned Kaggle notebooks.

---

## Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/rakeshg25ait2082/mlops_groupproject |
| Hugging Face Model | https://huggingface.co/Nlp0187/distilbert-imdb-reviews-v1 |
| Docker Image | https://hub.docker.com/repository/docker/ashwani0187/mlops-a3-inference |
| WandB Project | https://wandb.ai/ashwinmmmec794-iit-jodhpur/Mlops-Group-Project |
| Kaggle Notebook (v1) | https://www.kaggle.com/code/ashwanig20ait2022/mlops-project-transformers-model-imdb-v-1?scriptVersionId=325555945 |
| Kaggle Notebook (v2) | https://www.kaggle.com/code/ashwanig20ait2022/mlops-project-transformers-model-imdb-v-1?scriptVersionId=325561163 |

---

## Project Structure

```
├── Dockerfile
├── requirements.txt
├── id2label.json
└── src/
    └── inference.py
```

---

## How to Run

### Using Docker (recommended)

```bash
docker pull ashwani0187/mlops-a3-inference
docker run --rm -e INPUT_TEXT="This movie was amazing!" ashwani0187/mlops-a3-inference
```

### Locally

```bash
pip install -r requirements.txt
python src/inference.py
```

Set `INPUT_TEXT` env variable to pass custom text, otherwise a default sample is used.

---

## Experiment Results

Two versions were trained with different learning rates (3 epochs each).

### Version 1 (LR = 3e-5) — *Selected Version*

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|--------------|-----------------|----------|----------|
| 1 | 0.4661 | 0.4665 | 91.05% | 91.04% |
| 2 | 0.3299 | 0.4465 | 91.35% | 91.34% |
| 3 | 0.1500 | 0.5502 | 92.85% | 92.85% |

### Version 2 (LR = 1e-5)

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|--------------|-----------------|----------|----------|
| 1 | 0.5227 | 0.4595 | 90.80% | 90.80% |
| 2 | 0.4296 | 0.4554 | 90.90% | 90.89% |
| 3 | 0.2962 | 0.4552 | 91.70% | 91.70% |

### Summary

| Metric | Version 1 | Version 2 |
|--------|-----------|-----------|
| Learning Rate | 3e-5 | 1e-5 |
| Final Accuracy | **92.85%** | 91.70% |
| Final F1 Score | **92.85%** | 91.70% |

**Version 1** was selected for deployment — higher accuracy and F1 despite slightly higher validation loss. Version 2's lower learning rate led to slower convergence and would need more epochs to match.

---

## Tech Stack

- **Model**: DistilBERT (Hugging Face Transformers)
- **Dataset**: IMDB Reviews (10k train / 2k test sampled from 50k)
- **Tracking**: Weights & Biases
- **Serving**: Docker + Python inference script
- **Training**: Kaggle GPU notebooks