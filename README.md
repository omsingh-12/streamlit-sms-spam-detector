# 📩 Advanced SMS Spam Detection

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam). It trains and compares five classifiers on a combined dataset, saves the best pipeline, and provides an interactive **Streamlit** dashboard to explore the data, model reports, charts, and live predictions.

---


## ✨ Features

- Combines two SMS datasets into one larger training set
- Cleans data (removes duplicates and missing values)
- TF-IDF text vectorization (English stop-words removed, top 5,000 features)
- Compares 5 models: **Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost**
- Reports accuracy, F1 score, and full classification reports per model
- Saves the final pipeline (`TF-IDF + Multinomial Naive Bayes`) with `joblib`
- Streamlit web app with dataset viewer, reports, comparison charts, and custom predictions

---

## 📁 Project Structure

```
.
├── main.py              # Data loading, cleaning, model training, evaluation, saving
├── stream.py            # Streamlit dashboard
├── spam_pipeline.pkl    # Saved trained pipeline (TF-IDF + Naive Bayes)
├── requirements.txt     # Python dependencies
├── .gitignore
├── README.md
└── data/
    ├── SMSSpamCollection   # Dataset 1 (tab-separated: label, message)
    └── extra_spam.csv      # Dataset 2 (columns v1 = label, v2 = message)
```

> **Note:** The `data/` folder must contain both dataset files before running the project.

---

## 📊 Datasets

| Dataset | File | Format |
|---------|------|--------|
| Dataset 1 | `data/SMSSpamCollection` | Tab-separated, no header (`label`, `message`) |
| Dataset 2 | `data/extra_spam.csv` | CSV with `v1` (label) and `v2` (message) |
| Dataset 3 | Combined | Dataset 1 + Dataset 2 after cleaning |

The classic [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository and the commonly used Kaggle `spam.csv` are the sources of these formats.

---

## 🧠 How It Works

1. **Load** both datasets and merge them.
2. **Clean** — drop duplicates and null rows; map labels `ham → 0`, `spam → 1`.
3. **Split** — 80% train / 20% test (stratified, `random_state=42`).
4. **Vectorize** — `TfidfVectorizer(stop_words="english", max_features=5000)`.
5. **Train & compare** five classifiers and print classification reports.
6. **Save** the final pipeline (TF-IDF + Multinomial Naive Bayes) to `spam_pipeline.pkl`.
7. **Predict** — `custom_prediction()` loads the saved pipeline and scores sample messages with spam/ham probabilities.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the datasets

Place `SMSSpamCollection` and `extra_spam.csv` inside a `data/` folder in the project root.

### 5. Run the Streamlit app

```bash
streamlit run stream.py
```

The app opens at `http://localhost:8501`.

You can also run only the training script:

```bash
python main.py
```

---

## 🖥️ Dashboard Overview

- **Datasets** — view Dataset 1, Dataset 2, or the combined dataset along with their shapes
- **Classification Reports** — precision, recall, and F1 for all five models
- **Charts** — accuracy and F1-score comparison bar plots
- **Custom Prediction** — run the saved model on sample messages and see spam/ham probabilities

---

## 🛠️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- joblib
- Streamlit

---

## 🔮 Future Improvements

- Let users type their own message in the Streamlit app for live prediction
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Add n-gram features and text preprocessing (stemming/lemmatization)
- Add unit tests and a CI workflow
- Deploy on Streamlit Community Cloud

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.

---

## 🙋 Author

Created by **<Om Singh>** — [GitHub](https://github.com/omsingh-12)
