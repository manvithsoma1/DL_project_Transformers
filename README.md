# 🤖 Transformer-Based Text Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B.svg)](https://streamlit.io)
[![F1-Score](https://img.shields.io/badge/Macro_F1-0.54-brightgreen.svg)]()

A deep learning project comparing Custom Transformer Encoders against RNN baselines (SimpleRNN, LSTM, GRU, BiGRU) on the Reuters Newswire topic classification dataset. 

🚀 **Live Dashboard:** [Streamlit App](https://your-app-url.streamlit.app)

## 📊 Key Findings

We evaluated various sequential models designed to classify news articles into 46 distinct topics. A shallow 3-layer Transformer significantly outperformed complex RNN baselines on our ~9,000 sequence training set.

| Model | Macro F1-Score | Best Val Accuracy |
|--------|---------------|-------------------|
| **Transformer (3-Layer)** | **0.540** | **79.2%** |
| Transformer (5-Layer) | 0.523 | 77.2% |
| Transformer (7-Layer) | 0.411 | 75.0% |
| BiGRU | 0.386 | - |
| GRU | 0.179 | - |
| LSTM | 0.053 | - |

***Insight:*** *Deeper Transformers overfit quickly on smaller datasets. The 3-layer architecture served as the optimal capacity.*

## 🧠 Architecture Highlights
```text
[Input Sequence]
        │
   [Token + Positional Embedding]
        │
  ┌─────▼───────────────┐
  │ Transformer Encoder │ x3
  │ - Multi-Head Attn   │
  │ - FFN (32 -> 64)    │
  │ - GELU Activation   │
  └─────┬───────────────┘
        │
[Global Avg Pool] ++ [Global Max Pool]
        │
   [Dense Classifier]
```
- **Custom Token & Position Embedding:** Enables sequence order awareness.
- **Dual Pooling:** Concatenates Average and Max pooling for a richer global sequence representation.
- **GELU Activations:** Smoother gradients across FFN layers compared to standard ReLU.
- **Early Stopping & LR Decay:** Prevents overfitting during training.

## 🛠️ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/transformer-news-classification.git
cd transformer-news-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard locally
```bash
streamlit run app.py
```

### 4. Re-run Training (Optional)
To retrain the models and generate new results:
```bash
jupyter notebook notebooks/transformer_classification.ipynb
```

## 📚 What I Learned
- Designing custom Keras layers including positional embeddings.
- Implementing Transformer blocks from scratch using TensorFlow/Keras.
- Diagnosing overfitting using train-val accuracy gap analysis.
- Building interactive, data-driven web applications using Streamlit and Plotly.
