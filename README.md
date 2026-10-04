# 🛡️ CyberGuard AI — Cyberbullying Detection System using NLP and Machine Learning

A complete, production-quality **Final Year Cybersecurity Project** that detects cyberbullying,
toxic, abusive, offensive, hateful, and threatening messages in real time using classical NLP
preprocessing and supervised Machine Learning, wrapped in a polished Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

CyberGuard AI classifies any text message into one of six categories:

| Class | Description |
|---|---|
| ✅ **Safe** | Respectful, non-harmful communication |
| ⚠️ **Offensive** | Rude or disrespectful language |
| 🟠 **Toxic** | Language likely to make someone leave a discussion |
| 🔴 **Cyberbullying** | Targeted, repeated abusive behavior online |
| 🟣 **Hate Speech** | Attacks based on identity or group membership |
| ⛔ **Threat** | Explicit intent to cause harm |

For every prediction the system reports:
- **Confidence Score**
- **Risk Level** (None / Low / Medium / High / Critical)
- **Reason for Detection**
- **Highlighted harmful words**
- **Recommended Action**

Example:

```
Input: "You are useless and everyone hates you."

Prediction : Cyberbullying
Confidence : 97.49%
Risk       : Critical
Detected Words: useless, everyone hates you
Suggestion : This message contains abusive/bullying language. Avoid sending
             messages that may harm others. Consider reporting this if you received it.
```

---

## 🏗️ Architecture

```
                 ┌────────────────────┐
                 │   Streamlit UI      │
                 │  (app.py - 7 pages) │
                 └─────────┬───────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
     ┌─────────────┐ ┌────────────┐ ┌───────────────┐
     │ predict.py   │ │ utils/     │ │ utils/database │
     │ (inference)  │ │ preprocess │ │ (SQLite)       │
     └──────┬───────┘ └────────────┘ └───────────────┘
            │
            ▼
   ┌─────────────────────┐
   │ model/model.pkl      │  <-- trained by train_model.py
   │ model/tfidf.pkl       │
   │ model/label_encoder.pkl│
   └─────────────────────┘
```

**Training pipeline (`train_model.py`):**

```
CSV Dataset → Text Cleaning (utils/preprocessing.py) → TF-IDF →
  ┌── Logistic Regression
  ├── Naive Bayes
  ├── Linear SVM
  └── Random Forest
        → Compare accuracy/precision/recall/F1 → Select best → Save with Joblib
```

---

## 📂 Project Structure

```
CyberbullyingDetection/
│
├── app.py                  # Main Streamlit dashboard (7 pages)
├── train_model.py          # Trains & compares 4 ML models, saves the best one
├── predict.py               # Loads saved model and runs inference
├── setup_nltk.py            # One-time NLTK corpora downloader
├── gen_dataset.py            # Generates the bundled demo dataset
├── requirements.txt
├── README.md
│
├── dataset/
│   └── cyberbullying.csv    # Labeled training data (text,label)
│
├── model/
│   ├── model.pkl             # Best trained classifier
│   ├── tfidf.pkl              # Fitted TF-IDF vectorizer
│   ├── label_encoder.pkl      # Class label encoder
│   └── metrics.json           # Accuracy/precision/recall/F1 for all models
│
├── database/
│   └── cyberbullying.db       # SQLite database (auto-created)
│
├── utils/
│   ├── preprocessing.py       # NLP cleaning pipeline (NLTK + offline fallback)
│   ├── database.py             # SQLite access layer (parameterized queries)
│   ├── visualization.py        # Plotly + Matplotlib chart builders
│   └── helper.py                # Harmful-word lexicon, risk scoring, security helpers
│
├── static/                     # Static assets (optional, for future custom UI assets)
├── screenshots/                 # App screenshots for documentation
└── reports/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── model_comparison.csv
    └── app.log                  # Runtime activity log
```

---

## ⚙️ Installation

### 1. Clone / copy the project into VS Code

```bash
cd CyberbullyingDetection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP corpora (one time)

```bash
python setup_nltk.py
```

> If this step fails or you are offline, the app automatically falls back to a
> built-in lightweight preprocessing pipeline — nothing will crash.

### 5. Train the model

```bash
python train_model.py
```

This will:
- Load `dataset/cyberbullying.csv`
- Clean and vectorize the text (TF-IDF)
- Train Logistic Regression, Naive Bayes, Linear SVM, and Random Forest
- Automatically select the best model by accuracy
- Save `model/model.pkl`, `model/tfidf.pkl`, `model/label_encoder.pkl`
- Save evaluation charts to `reports/`

### 6. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

### 7. (CLI) Test predictions directly

```bash
python predict.py "You are useless and everyone hates you."
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Overview, quick stats, class legend |
| 📖 **About Project** | Objective, pipeline, tech stack |
| 🔍 **Cyberbullying Detection** | Live text analysis with confidence gauge, harmful-word highlighting, live typing warnings, speech-to-text upload |
| 📊 **Statistics** | Pie/bar charts, timeline, top offensive words |
| 🕘 **History** | Searchable/filterable prediction log with CSV export |
| 🔐 **Admin Dashboard** | Password-protected panel: totals, top words, CSV report, delete history |
| 🎯 **Model Accuracy** | Model comparison table/chart, confusion matrix, ROC curve |

The sidebar includes a **🌙 Dark Mode** toggle and displays the active NLP backend.

---

## 🧠 Machine Learning Details

**Preprocessing:** lowercase → remove URLs → remove HTML → remove numbers → remove emojis →
remove punctuation → tokenize → remove stopwords → lemmatize.

**Feature Extraction:** TF-IDF (unigrams + bigrams, 5000 max features).

**Models compared** (in `train_model.py`):
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM (calibrated for probability output)
- Random Forest

**Metrics tracked:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC/AUC (one-vs-rest).

> **Note on the bundled dataset:** `dataset/cyberbullying.csv` is a synthetically generated,
> template-based demonstration dataset (1,584 rows / 6 classes) included so the project runs
> immediately out of the box. Because it is template-based, benchmark accuracy on it is
> artificially high (~100%). **For a real academic submission or production use, replace it
> with a real-world corpus** (e.g. the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
> or a Kaggle cyberbullying dataset) using the same `text,label` column format, then re-run
> `python train_model.py`.

---

## 🔐 Cybersecurity Features

- **Input Validation** — empty/oversized input rejected (`utils/helper.py::validate_input`)
- **SQL Injection Protection** — all database queries use parameterized placeholders (`utils/database.py`); raw input is never string-formatted into SQL
- **XSS Protection** — user input is HTML-escaped and script tags stripped before storage/display (`utils/helper.py::sanitize_input`)
- **Suspicious Pattern Flagging** — regex-based detection of classic SQLi/XSS payloads with in-app warnings
- **Rate Limiting** — in-memory sliding-window limiter caps predictions per session (default: 20/minute)
- **Logging** — all predictions and admin actions are logged to `reports/app.log`
- **Admin Authentication** — password-gated admin dashboard (override via `ADMIN_PASSWORD` env var)

---

## ✨ Bonus Features Included

- ⚡ **Live typing detection** — instant keyword warning while the user types, before clicking Analyze
- 🎙️ **Speech-to-text detection** — optional audio upload + transcription (requires `SpeechRecognition`)
- 🖍️ **Profanity word highlighting** — harmful terms highlighted inline in the detection report
- 📄 **CSV export** — History and Admin pages support full/filtered CSV report downloads
- 🌙 **Dark mode** — toggle in the sidebar
- 🧩 **Offline-safe NLP** — automatic fallback pipeline if NLTK corpora aren't downloaded

> Multi-language support, PDF export, and email alerts are stubbed as **optional dependencies**
> in `requirements.txt` (commented out) — install `langdetect`, `fpdf2`, and configure SMTP
> credentials respectively to activate them as an extension exercise.

---

## 📊 Sample Results

Run `python train_model.py` to regenerate. Example run on the bundled demo dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 100.00% | 100.00% | 100.00% | 100.00% |
| Naive Bayes | 100.00% | 100.00% | 100.00% | 100.00% |
| Linear SVM | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% |

*(Scores are near-perfect because the bundled demo dataset is template-generated; see note above.)*

Confusion matrix and ROC curve images are saved to `reports/confusion_matrix.png` and
`reports/roc_curve.png`, and are also viewable inside the app's **Model Accuracy** page.

---

## 🚀 Future Scope

- Retrain on a large real-world dataset (Jigsaw / Kaggle) for production-grade generalization
- Upgrade to deep learning (LSTM, BiLSTM, or a fine-tuned BERT/DistilBERT transformer)
- Add multi-language toxicity detection using `langdetect` + translated pipelines
- Deploy as a browser extension / REST API for real-time moderation on live chat platforms
- Add email/SMS alerting to moderators for Critical-risk messages
- Add user authentication and per-user history instead of session-based tracking
- Containerize with Docker and deploy to cloud (AWS/GCP/Azure) with a managed database

---

## 🖼️ Screenshots

Add your own screenshots of the running app to the `screenshots/` folder, for example:
- `screenshots/home.png`
- `screenshots/detection.png`
- `screenshots/statistics.png`
- `screenshots/admin.png`
- `screenshots/model_accuracy.png`

---

## 📜 License

This project is provided for academic/educational purposes as a final-year cybersecurity
project template. Feel free to adapt it with attribution.

---

## 🙌 Acknowledgements

- Scikit-learn, NLTK, spaCy, Streamlit, Plotly, and Matplotlib open-source communities
- Jigsaw / Kaggle Toxic Comment Classification Challenge for dataset inspiration
