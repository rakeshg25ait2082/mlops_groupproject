# MLOps Group Project — End-to-End Sentiment Analysis Pipeline using DistilBERT

## Project Overview

This project implements a complete end-to-end MLOps pipeline for sentiment analysis using the IMDB Movie Reviews dataset and the DistilBERT transformer model from Hugging Face. The pipeline demonstrates data preparation, model training, experiment tracking, model deployment, containerized inference, and workflow automation following industry-standard MLOps practices.

The objective was not only to achieve strong classification performance but also to build a reproducible, scalable, and automated machine learning workflow.

---

## Project Links

| Resource                    | URL                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| GitHub Repository           | https://github.com/rakeshg25ait2082/mlops_groupproject                                                            |
| Hugging Face Model          | https://huggingface.co/Nlp0187/distilbert-imdb-reviews-v1                                                         |
| Docker Image                | https://hub.docker.com/r/ashwani0187/mlops-a3-inference                                                           |
| Weights & Biases Dashboard  | https://wandb.ai/ashwinmmmec794-iit-jodhpur/Mlops-Group-Project                                                   |
| Kaggle Notebook (Version 1) | https://www.kaggle.com/code/ashwanig20ait2022/mlops-project-transformers-model-imdb-v-1?scriptVersionId=325555945 |
| Kaggle Notebook (Version 2) | https://www.kaggle.com/code/ashwanig20ait2022/mlops-project-transformers-model-imdb-v-1?scriptVersionId=325561163 |

---

## Technology Stack

| Component            | Technology                             |
| -------------------- | -------------------------------------- |
| Model                | DistilBERT base uncased(Hugging Face Transformers) |
| Dataset              | IMDB Movie Reviews                     |
| Training Platform    | Kaggle GPU Notebooks                   |
| Experiment Tracking  | Weights & Biases (W&B)                 |
| Model Registry       | Hugging Face Hub                       |
| Containerization     | Docker                                 |
| CI/CD Automation     | GitHub Actions                         |
| Programming Language | Python 3.11                            |

---

## Dataset Information

### Dataset

IMDB Movie Reviews Dataset

### Dataset Size

* Original Dataset: 50,000 reviews
* Training Sample: 10,000 reviews
* Testing Sample: 2,000 reviews

### Classification Task

Binary Sentiment Classification

Classes:

* Positive (1)
* Negative (0)

### Data Preparation and Normalisation

The dataset was inspected for quality issues and prepared for model training using a reusable preprocessing pipeline.

Preprocessing steps included:

* Removal of missing or invalid records, html tags, punctuation marks
* Text normalization and cleanup
* Label encoding
* Tokenization using the DistilBERT tokenizer
* Generation of label mappings for deployment

### Label Mapping

```json
{
  "0": "negative",
  "1": "positive"
}
```

---

## Model Selection

DistilBERT base uncased was selected because it provides an excellent balance between performance and efficiency. It retains approximately 97% of BERT's language understanding capabilities while reducing model size and inference time significantly. The lightweight architecture makes it suitable for Kaggle GPU training, Docker deployment, and GitHub Actions-based inference workflows. Its strong performance on text classification tasks makes it an ideal choice for sentiment analysis on the IMDB dataset.

---

## Repository Structure

```text
mlops_groupproject/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── inference.yml
│
├── report/
│   └── MLOps_FinalGroupReport_Group1.pdf
|
├── src/
│   └── inference.py
│
├── id2label.json
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Model Training

Training was performed using Kaggle GPU notebooks with experiment tracking enabled through Weights & Biases.

Two experiment versions were executed with different learning rates.

### Version 1

* Learning Rate: 3e-5
* Epochs: 3

### Version 2

* Learning Rate: 1e-5
* Epochs: 3

All metrics, hyperparameters, losses, accuracy, and F1 scores were logged automatically to Weights & Biases.

---

## Experiment Results

### Version 1 (Selected Model)

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.4661        | 0.4665          | 91.05%   | 91.04%   |
| 2     | 0.3299        | 0.4465          | 91.35%   | 91.34%   |
| 3     | 0.1500        | 0.5502          | 92.85%   | 92.85%   |

### Version 2

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.5227        | 0.4595          | 90.80%   | 90.80%   |
| 2     | 0.4296        | 0.4554          | 90.90%   | 90.89%   |
| 3     | 0.2962        | 0.4552          | 91.70%   | 91.70%   |

### Comparison Summary

| Metric         | Version 1 | Version 2 |
| -------------- | --------- | --------- |
| Learning Rate  | 3e-5      | 1e-5      |
| Final Accuracy | 92.85%    | 91.70%    |
| Final F1 Score | 92.85%    | 91.70%    |

### Selected Model

Version 1 was selected for deployment because it achieved the highest Accuracy and F1 Score. Although Version 2 produced a slightly lower validation loss, Version 1 demonstrated superior classification performance on the validation dataset.

---

## Running Inference Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference:

```bash
python src/inference.py
```

Set a custom input:

```bash
export INPUT_TEXT="This movie was amazing!"
python src/inference.py
```

---

## Docker Deployment

### Pull Docker Image

```bash
docker pull ashwani0187/mlops-a3-inference
```

### Run Container

```bash
docker run --rm -e INPUT_TEXT="This movie was amazing!" ashwani0187/mlops-a3-inference
```

The Docker image automatically loads the deployed Hugging Face model and performs sentiment classification on the supplied text.

---

## GitHub Actions

Two GitHub Actions workflows were implemented.

### CI Workflow

Triggered on:

* Push to develop branch
* Pull request to main branch

Functions:

* Install dependencies
* Run linting checks
* Validate repository integrity

### Inference Workflow

Triggered manually using workflow_dispatch.

Functions:

* Load the deployed Hugging Face model
* Accept user input text
* Execute sentiment prediction
* Display classification output

---

## Team Contributions

| Team Member            | Contribution                                            |
| -----------------------| ------------------------------------------------------- |
| Ashwani Bhardwaj       | Performed data collection and preprocessing, prepared the IMDB dataset, implemented and fine-tuned the DistilBERT model on Kaggle GPU, conducted multiple training experiments, tracked metrics using Weights & Biases (W&B), and deployed the final model to Hugging Face Hub.  |
| Jeenal Chaudhary       | Developed and configured GitHub Actions workflows (ci.yml and inference.yml), performed workflow testing and validation, updated and enhanced project documentation including the README, contributed to end-to-end pipeline testing, and prepared the final project report. |
| Rakesh Panda           | Managed GitHub repository administration, configured collaborators and access permissions, implemented branch protection policies, maintained repository structure, and supported GitHub workflow integration and project coordination.               |
| Harshith Duggisetty    | Conducted initial data collection and dataset exploration, participated in model selection and evaluation, created the initial project README, developed the Docker containerization setup, and assisted with deployment-related tasks.                   |

---

## Conclusion

This project successfully demonstrates a complete MLOps lifecycle including dataset preparation, transformer fine-tuning, experiment tracking, model deployment, containerization, and CI/CD automation. The deployed DistilBERT model achieved 92.85% accuracy on the IMDB sentiment classification task while maintaining a reproducible and production-oriented workflow.
