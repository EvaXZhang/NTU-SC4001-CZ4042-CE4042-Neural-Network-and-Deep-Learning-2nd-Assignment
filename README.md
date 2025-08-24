# Financial Sentiment Analysis with Deep Learning


## Table of Contents
- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Notes & Limitations](#notes--limitations)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project explores **sentiment analysis on financial text** with an emphasis on **small datasets** and **vocabulary sparsity**. We benchmarked four neural models — **LSTM**, **Attention-only**, **LSTM+Attention**, and **Transformer+Attention** — on the **FinancialPhraseBank** dataset. All deep models showed **overfitting** when trained from scratch.

We then introduced a **vocabulary-aware preprocessing** pipeline (replace low-frequency tokens with `<UNK>` + stemming) and trained a **simple MLP** on averaged embeddings. The **MLP** achieved **higher validation accuracy** and **better stability**, with no observable overfitting.

---

## Key Contributions
- Diagnose **vocabulary sparsity** in small financial text data (9,255 unique words; many rare/unseen in train vs. valid).
- Replace **2,717** low-frequency/validation-only tokens with `<UNK>`; apply **Porter stemming**.
- Demonstrate that a **compact MLP** over averaged embeddings (embed_dim=200, hidden_dim=200, dropout=0.2) **outperforms** heavier sequence models under low-resource settings.

---

## Repository Structure
```text
├── data/                 # FinancialPhraseBank dataset (download separately)
├── notebooks/            # EDA, preprocessing, training & evaluation notebooks
├── models/               # Model definitions (LSTM, Attention, LSTM+Attn, Transformer+Attn, MLP)
├── utils/                # Preprocessing, dataset class, tokenization, plotting helpers
├── results/              # Logs, metrics, curves, comparison figures, saved models
├── train.py              # CLI entry to train/evaluate models
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup
```bash
# 1) Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2) Python env (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt
```

**Main dependencies:** Python 3.9+, PyTorch, Pandas, NumPy, NLTK, Matplotlib, scikit-learn, tqdm.

---

## Dataset
We use **FinancialPhraseBank** (Malo et al., 2014): financial news sentences labeled **positive / negative / neutral**.

- Labels used here: **0 = Neutral, 1 = Negative, 2 = Positive**.
- Download (per license) and place files under `data/`.

> Note: The dataset offers multiple agreement levels (e.g., `Sentences_AllAgree`, `Sentences_50Agree`). Use one consistently, or combine with care.

---

## Preprocessing
Steps applied:
- **Lowercasing & punctuation removal** (regex).
- **Tokenization & English stopword removal** (NLTK).
- **Digit filtering** (drop tokens with numbers).
- **Label encoding**: neutral→0, negative→1, positive→2.
- **Vocabulary build** from training text only.
- **Low-frequency handling**: identify **2,717** low/val-only words ⇒ replace with `<UNK>`.
- **Stemming**: **PorterStemmer** to reduce inflectional variants.

**Data split:** first **2,500** samples as **train**, the rest as **validation** (to replicate our analysis).  
**Padding:** original runs used **batch_size=1** to avoid sequence padding; you may switch to padding + collate for speed.

---

## Models
Baseline models:
1. **LSTM** — embedding → LSTM → final hidden → MLP classifier.  
2. **Attention-only** — embedding → token-wise transform + attention weights → weighted sum → classifier.  
3. **LSTM + Attention** — LSTM outputs + token-level attention → pooled vector → classifier.  
4. **Transformer + Attention** — learnable positional embedding → TransformerEncoder → secondary attention → classifier.

Proposed model:
- **MLP over averaged embeddings**  
  - **Embed dim**: 200  
  - **Hidden dim**: 200  
  - **Dropout**: 0.2  
  - **Pooling**: mean over token embeddings  
  - **Head**: Linear → ReLU → Dropout → Linear (to num_classes)

---

## Training & Evaluation

### Quick Start
Preprocess:
```bash
python utils/preprocess.py \
  --data_dir data \
  --output_dir data/processed \
  --min_freq 2 \
  --apply_stemmer
```

Train (choose one):
```bash
# LSTM
python train.py --model lstm --epochs 10 --batch_size 1 --embed_dim 200 --hidden_dim 200 --dropout 0.2

# Attention-only
python train.py --model attention --epochs 10 --batch_size 1 --embed_dim 200 --hidden_dim 200 --dropout 0.2

# LSTM + Attention
python train.py --model lstm_attn --epochs 10 --batch_size 1 --embed_dim 200 --hidden_dim 200 --dropout 0.2

# Transformer + Attention
python train.py --model transformer_attn --epochs 10 --batch_size 1 --embed_dim 200 --hidden_dim 200 --dropout 0.2 --n_heads 4 --n_layers 2

# MLP (improved)
python train.py --model mlp --epochs 10 --batch_size 32 --embed_dim 200 --hidden_dim 200 --dropout 0.2
```

Common flags (if supported in your script):
```bash
--lr 1e-3 --weight_decay 1e-4 --max_len 128 --seed 42 --device cuda
```

Visualize:
```bash
python utils/plot_results.py --results_dir results
```

Saved artifacts:
- `results/*/metrics.json` — train/val loss & accuracy
- `results/*/curves.png` — learning curves
- `results/*/model.pt` — best checkpoint

---

## Results
**Observation across baselines:** training loss ↓ steadily; **validation loss stagnates or worsens**; validation accuracy **unstable** ⇒ **overfitting**.

**Attention-based** approaches are relatively more stable; **Transformer+Attention** shows consistent validation but still limited by data scale.

**Proposed MLP** (with vocab-aware preprocessing):
- **No overfitting** observed; **test accuracy ≥ train accuracy** at times.
- **Test loss** continues to drop without spikes/plateaus.
- **Best overall validation performance** among all models.

| Model                    | Validation Accuracy (trend) | Notes                                    |
|-------------------------|-----------------------------|------------------------------------------|
| LSTM                    | Low / unstable              | Memorizes quickly; poor generalization   |
| Attention-only          | Slightly better             | More robust to sparsity                   |
| LSTM + Attention        | Overfits                    | Gains context, but still unstable         |
| Transformer + Attention | Most consistent baseline    | Underutilized under small data            |
| **MLP (ours)**          | **Best**                    | Stable, no overfitting, simplest          |

> Numbers depend on random seed and exact split; see `results/` for your run logs and plots.

---

## Reproducibility
- Fix seeds: `--seed 42` (and set PyTorch/CUDA deterministic flags if needed).
- Log everything under `results/<run_name>/`.
- Record: model config, preprocessing switches, dataset split checksum.
- Provide environment export (e.g., `pip freeze > results/requirements-freeze.txt`).

---

## Notes & Limitations
- Using **batch_size=1** avoids padding issues but is slow; consider padding + attention mask for speed.
- With more data or pre-trained embeddings (e.g., GloVe/fastText), baselines may improve.
- Domain-specific pretraining (FinBERT, etc.) is intentionally **out of scope** to keep the focus on **small-data-from-scratch**.

---

## Citation
If you use the dataset, please cite:

```bibtex
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Malo, Pekka and Sinha, Ankur and Korhonen, Pyry and Wallenius, Jyrki and Takala, Petri},
  journal={Journal of the Association for Information Science and Technology},
  volume={65},
  year={2014}
}
```

---

## Acknowledgements
- Course: **SC4001 CE/CZ4042 – Neural Networks and Deep Learning**
- Dataset: **FinancialPhraseBank** (Malo et al., 2014)

---
