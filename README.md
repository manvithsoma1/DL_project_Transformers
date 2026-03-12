# 🧠 Transformer Encoder for News Classification

<p align="center">
  <a href="https://dlprojecttransformers-huejodfutwmxmecfpf3tvs.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Best%20Macro%20F1-0.54-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Val%20Accuracy-79.2%25-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Beats%20BiGRU-+40%25%20F1-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20Cloud-ff4b4b?style=flat-square&logo=streamlit" />
</p>

<p align="center">
  <b>🔴 <a href="https://dlprojecttransformers-huejodfutwmxmecfpf3tvs.streamlit.app/">Live Demo — Try It Now</a></b>
</p>

<p align="center">
  A custom Transformer Encoder architecture built from scratch to classify Reuters news articles
  into <b>46 topic categories</b>. Systematically studies the effect of model depth (3 / 5 / 7 encoder layers)
  and benchmarks against 6 RNN baselines — proving Transformers dominate on this NLP task.
</p>

---

## 🚀 Live Dashboard

👉 **[https://dlprojecttransformers-huejodfutwmxmecfpf3tvs.streamlit.app/](https://dlprojecttransformers-huejodfutwmxmecfpf3tvs.streamlit.app/)**

The interactive dashboard includes:
- 📊 **Overview** — KPIs, F1 comparison, key findings
- 📈 **Training Analysis** — Accuracy & loss curves, overfitting analysis
- 🔬 **Model vs Baselines** — All 9 models compared (6 RNN + 3 Transformer)
- 🏗️ **Architecture Deep Dive** — Design decisions, hyperparameters, diagrams

---

## 📊 Experiment Results

| Model | Macro F1 | Best Val Accuracy | Parameters |
|-------|----------|-------------------|------------|
| 🏆 **Transformer — 3 Layers** | **0.540** | **79.2%** | 380,718 |
| Transformer — 5 Layers | 0.523 | 77.2% | 402,818 |
| Transformer — 7 Layers | 0.411 | 75.0% | 424,918 |
| BiGRU *(best RNN)* | 0.386 | — | — |
| GRU | 0.179 | — | — |
| SimpleRNN / LSTM | ~0.053 | — | — |
| BiSimpleRNN / BiLSTM | ~0.008 | — | — |

> **Key insight:** 3-Layer Transformer is optimal — deeper models overfit on ~9K training samples.
> All Transformer configs beat the best RNN baseline (BiGRU) by a significant margin.

---

## 🔥 Key Findings

```
1. 🏆 3-Layer Transformer  →  Best F1 (0.540) + least overfitting
2. 📉 Depth hurts here     →  7-layer degrades to F1=0.41 (overfitting)
3. 🚀 Transformers >> RNNs →  +40% F1 improvement over best RNN (BiGRU)
4. ⚡ Fast convergence     →  Best val accuracy reached by epoch 2-3
5. ⚠️  Class imbalance     →  "earn" class dominates, suppresses macro F1
```

---

## 🏗️ Architecture

```
Input Tokens (maxlen=200)
        │
        ▼
┌───────────────────┐
│  Token Embedding  │  vocab_size=10,000 → embed_dim=32
│  + Position Emb   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Encoder Block 1  │  ff_dim = 32  ← lightweight entry
│  (first)          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Encoder Blocks   │  ff_dim = 64  ← wider capacity
│  (intermediate)   │  (repeated N-2 times)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Encoder Block N  │  ff_dim = 32  ← lightweight exit
│  (last)           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ GlobalAvgPool1D   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Dense(46, softmax)│  → 46 topic classes
└───────────────────┘
```

**Each Encoder Block:**
```
Input
  │
  ├──→ MultiHeadAttention (4 heads) ──→ Dropout ──→ Add ──→ LayerNorm
  │                                                           │
  └──────────────────────────────────────────────────────────┘
                                                              │
  ┌───────────────────────────────────────────────────────────┘
  │
  ├──→ FFN: Dense(ff_dim, relu) → Dense(embed_dim) ──→ Dropout ──→ Add ──→ LayerNorm
  │
  └── Output (same shape as input — enables residual connections)
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `vocab_size` | 10,000 | Top-k most frequent words |
| `maxlen` | 200 | Covers ~90% of articles |
| `embed_dim` | 32 | Fixed across all blocks (residual compatibility) |
| `num_heads` | 4 | 4 × 8-dim attention heads |
| `ff_dim1` | 32 | First & last blocks — lightweight |
| `ff_dim2` | 64 | Intermediate blocks — wider capacity |
| `dropout` | 0.1 | Regularization |
| `optimizer` | Adam (lr=1e-3) | Adaptive learning rate |
| `loss` | Categorical Cross-Entropy | Multi-class classification |

---

## 📁 Project Structure

```
transformer-news-classification/
├── 📓 notebooks/
│   └── transformer_classification.ipynb  ← full experiment notebook
├── 📊 reports/
│   ├── eda.png
│   ├── training_curves.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── depth_analysis.png
├── 🤖 models/
│   ├── best_model.keras
│   ├── results.json
│   └── word_index.json
├── app.py                ← Streamlit dashboard
├── requirements.txt
├── runtime.txt           ← python-3.11
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/transformer-news-classification.git
cd transformer-news-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook (trains all 3 model configs)
jupyter notebook notebooks/transformer_classification.ipynb

# 4. Launch the dashboard
streamlit run app.py
```

---

## 📈 Training Curves (3-Layer — Best Model)

```
Epoch  │  Train Acc  │  Val Acc   │  Note
───────┼─────────────┼────────────┼──────────────────────
  1    │   62.3%     │   75.9%    │ Fast initial learning
  2    │   80.9%     │   79.1%    │ 
  3    │   87.8%     │   79.2%    │ ← Peak val accuracy
  4    │   91.8%     │   76.4%    │ Overfitting begins
  5    │   93.9%     │   73.8%    │
  10   │   95.6%     │   73.8%    │ Train >> Val → overfit
```

> Training accuracy kept climbing to 95%+ while validation accuracy peaked at epoch 3.
> This classic overfitting pattern motivates EarlyStopping in the optimized version.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `TensorFlow / Keras` | Model building & training |
| `scikit-learn` | F1-score, confusion matrix |
| `matplotlib / seaborn` | Training visualization |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Live web dashboard |

---

## 🧠 What I Learned

- **Encoder-only is enough for classification** — the decoder is only needed for sequence generation; for learning representations, the encoder alone is more efficient
- **Variable FFN width is a smart design choice** — intermediate blocks benefit from wider FFN (64 vs 32) while keeping first/last blocks lightweight for stability
- **Small datasets punish deep models** — with only ~9K samples, a 7-layer model has too many parameters and overfits aggressively; 3 layers is the sweet spot
- **Transformers generalize better than RNNs** — even without pretraining, a simple 3-layer transformer from scratch beats BiGRU by 40% on macro F1
- **Class imbalance dominates the metric** — "earn" class has 10× more samples than most classes; this suppresses macro F1 even when accuracy looks decent
- **Positional embedding is critical** — without it, the model has no notion of word order, which significantly degrades sequence understanding

---

## 📬 Connect

If you found this useful, please ⭐ star the repo!

Built as part of a Deep Learning course — custom Transformer from scratch, no pre-trained models used.

---

<p align="center">
  <a href="https://dlprojecttransformers-huejodfutwmxmecfpf3tvs.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Dashboard-Click%20Here-ff4b4b?style=for-the-badge" />
  </a>
</p>
