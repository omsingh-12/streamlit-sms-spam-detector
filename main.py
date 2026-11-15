import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

# -------------------------------
# 1. LOAD BOTH DATASETS
# -------------------------------
data1 = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin1"
)

data2 = pd.read_csv("data/extra_spam.csv", encoding="latin1")
data2 = data2[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

data = pd.concat([data1, data2], ignore_index=True)
shape1 = data1.shape
shape2 = data2.shape
shape3 = data.shape

# -------------------------------
# 2. CLEAN DATA
# -------------------------------
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label_num"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------
# 3. DEFINE MODELS FOR COMPARISON
# -------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# results = []

# for name, clf in models.items():
#     pipeline = Pipeline([
#         ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
#         ("clf", clf)
#     ])
#     pipeline.fit(x_train, y_train)
#     y_pred = pipeline.predict(x_test)
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
#     report =f"\n{name} Report:\n", classification_report(y_test, y_pred)
#     print(report)

reports = {}
results = []

for name, clf in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", clf)
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
    print(f"\n{name} Report:\n", classification_report(y_test, y_pred))

    # Save classification report
    reports[name] = classification_report(y_test, y_pred)
    
# -------------------------------
# 4. VISUALIZE MODEL COMPARISON
# -------------------------------
results_df = pd.DataFrame(results)
# def fig1():
#    plt.figure(figsize=(10, 6))
#    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Blues")
#    plt.title("Model Accuracy Comparison")
#    plt.ylim(0.9, 1.0)
#    plt.show()

# def fig2():
#    plt.figure(figsize=(10, 6))
#    sns.barplot(x="Model", y="F1 Score", data=results_df, palette="Greens")
#    plt.title("Model F1 Score Comparison")
#    plt.ylim(0.9, 1.0)
#    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

def fig1():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Blues", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0.9, 1.0)
    return fig   # ✅ return the figure object

def fig2():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="F1 Score", data=results_df, palette="Greens", ax=ax)
    ax.set_title("Model F1 Score Comparison")
    ax.set_ylim(0.9, 1.0)
    return fig   # ✅ return the figure object


fig1()
fig2()
# -------------------------------
# 5. SAVE BEST MODEL 
# -------------------------------
best_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("nb", MultinomialNB(alpha=1.0))
])
best_pipeline.fit(x_train, y_train)
joblib.dump(best_pipeline, "spam_pipeline.pkl")

# -------------------------------
# 6. TEST CUSTOM MESSAGES
# -------------------------------

def custom_prediction():
    loaded_pipeline = joblib.load("spam_pipeline.pkl")

    messages = [
        "Congratulations! You have won a free lottery prize. Click now!",
        "Hey, are we meeting today?",
        "URGENT! You have won a cash prize. Call now to claim your reward.",
        "Can you send me the assignment?"
    ]    

    predictions = loaded_pipeline.predict(messages)
    probabilities = loaded_pipeline.predict_proba(messages)

    results = []
    for msg, pred, prob in zip(messages, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        results.append({
            "Message": msg,
            "Prediction": label,
            "Spam Probability": f"{prob[1]*100:.2f}%",
            "Ham Probability": f"{prob[0]*100:.2f}%"
        })
    return results 