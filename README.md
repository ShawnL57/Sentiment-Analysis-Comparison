# Sentiment Analysis Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quick-start)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)

> **Compare the speed & accuracy of a classic lexicon‑based approach (VADER) with a modern transformer (RoBERTa) on 1 000 Amazon product reviews.**

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Example](#usage-example)
- [Results](#results)
- [Interpretations](#interpretations)
- [Extending the Work](#extending-the-work)
- [Dataset](#dataset)

---

## Features

- **Two sentiment engines**: VADER (`nltk`) & Twitter‑RoBERTa (`cardiffnlp`).
- **Visual diagnostics**: scatter‑matrix and classification metrics.
- **GPU‑optional**: VADER runs on CPU; RoBERTa can leverage GPU.
- **Pluggable**: drop in other models by editing one config dictionary.

---

## Project Structure

```text
.
├── Sentiment Analysis Comparison.ipynb  # Jupyter notebook — full experiment
├── reviews.csv                          # 1 000 sample product reviews
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your‑username>/sentiment‑analysis‑comparison.git
cd sentiment‑analysis‑comparison

# 2. Create & activate a virtual environment (Python ≥ 3.9)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon  # one‑off for VADER

# 4. Launch Jupyter & explore
jupyter notebook Sentiment\ Analysis\ Comparison.ipynb
```

---

## Usage Example

```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

reviews = pd.read_csv("reviews.csv", nrows=1000)
texts = reviews["Text"].tolist()

# --- VADER ---
vad = SentimentIntensityAnalyzer()
vader_scores = [vad.polarity_scores(t)["compound"] for t in texts]

# --- RoBERTa ---
name = "cardiffnlp/twitter-roberta-base-sentiment"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()

with torch.no_grad():
    logits = []
    for batch in [texts[i:i+32] for i in range(0, len(texts), 32)]:
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
        out = model(**inputs)
        logits.append(out.logits)
    logits = torch.cat(logits, dim=0)
    probs = torch.nn.functional.softmax(logits, dim=1)  # [:, 2] = positive prob
roberta_pos = probs[:, 2].tolist()

print(f"VADER avg compound: {sum(vader_scores)/len(vader_scores):.3f}")
print(f"RoBERTa avg positive prob: {sum(roberta_pos)/len(roberta_pos):.3f}")
```

---

## Results

| Metric       | VADER (thresh > 0.05) | RoBERTa |
|--------------|-----------------------|---------|
| **Accuracy** | ~0.63                 | ~0.78   |
| **Micro‑F1** | ~0.59                 | ~0.77   |

> *Numbers will vary slightly — rerun to reproduce.*

---

## Interpretations

- **Speed vs. Accuracy**  
  - VADER is CPU‑cheap and suitable for real‑time use.  
  - RoBERTa is heavier but captures nuance.
- **Domain Robustness**  
  - Lexicon methods struggle with domain‑specific slang, whereas transformer models can capture it after fine‑tuning.

---

## Extending the Work

- Fine‑tune RoBERTa on domain‑specific data.
- Swap in DistilBERT, DeBERTa, or LLaMA 2 adapters.
- Add k‑fold cross‑validation and hyper‑parameter sweeps.
- Wrap as a Streamlit or FastAPI micro‑service.

---

## Dataset

The sample `reviews.csv` is a subset of the **Amazon Fine Food Reviews** corpus — [McAuley & Leskovec 2013](https://snap.stanford.edu/data/web-FineFoods.html).

---

