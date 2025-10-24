# 🧠 Toxic Comment Classification (SNLP Project)

This project builds a **toxic comment classifier** using both **traditional machine learning** (Logistic Regression) and **deep learning** (LSTM). It is part of the Social Natural Language Processing (SNLP) project and uses the Jigsaw Toxic Comment dataset.

---

## 🚀 Features

- Text preprocessing and cleaning (stopword removal, punctuation stripping, lowercasing)
- Exploratory Data Analysis (class distribution, toxic word cloud)
- Feature extraction:
  - TF-IDF for Logistic Regression
  - Tokenization & Padding for LSTM
- Model Training:
  - Logistic Regression for the “toxic” label
  - LSTM for multi-label classification
- Model Evaluation with confusion matrices and classification reports
- Prediction on new comments
- Model saving (`.pkl`, `.keras` formats)

---

## 📂 Project Structure

```
toxic-comment-classifier/
│
├── README.md
├── requirements.txt
├── snlp_project.py
├── SNLP_logistic_regression.ipynb
├── SNLP_LSTM.ipynb
└── models/
```

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

1. Place your dataset (`train.csv`) in the same directory.
2. Run in Colab or locally:
   ```bash
   python snlp_project.py
   ```
3. The script trains both models and saves them in `/models`.

---

## 🧪 Example Prediction

```python
from snlp_project import predict_toxicity

new_comment = "You are an amazing person!"
predict_toxicity(new_comment, lr_model, dl_model, tfidf_vectorizer, tokenizer, max_sequence_length, toxicity_classes)
```

---

## 📊 Results Summary

| Model | Focus | Strengths | Weaknesses |
|--------|--------|------------|-------------|
| Logistic Regression | ‘toxic’ label | Simple, interpretable | Limited on multi-label |
| LSTM | All labels | Captures context | Struggles with rare classes |

---

## 🧩 Next Steps
- Handle data imbalance using SMOTE or class weights
- Try Bidirectional LSTM or transformer models (e.g., BERT)
- Experiment with threshold tuning and hyperparameter optimization

---

**Author:** Tk Vp  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
