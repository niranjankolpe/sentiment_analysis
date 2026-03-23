import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes  import MultinomialNB
from sklearn.svm          import LinearSVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

import joblib

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from text_preprocessing import preprocess, preprocess_label

import os

models_path = "models/"
os.makedirs(models_path, exist_ok=True)

reports_path = "reports/"
os.makedirs(reports_path, exist_ok=True)

df = pd.read_csv("data/sentimentdataset.csv")
# print(df.head())
# print(df.iloc[0])

# print(df.isnull().sum())

df = df[["Text", "Sentiment"]]
# print(df.head())

df.Text = df.Text.apply(preprocess)
df.Sentiment = df.Sentiment.apply(preprocess_label)
# print(df.head())

# print(df.Sentiment.value_counts().keys())

positive = [
    "Positive","Joy","Excitement","Elation","Inspired","Enthusiasm",
    "Serenity","Fulfillment","Contentment","Euphoria","Grateful",
    "Awe","Proud","Hopeful","Happiness","Inspired"
]

negative = [
    "Desolation","Hate","Betrayal","Loneliness","Fearful","Grief",
    "Frustration","Heartbreak","Melancholy","Despair","Bitterness",
    "Exhaustion","Bad","Bitter","Resentment"
]

neutral = ["Neutral","Confusion","Curiosity","Ambivalence"]

def convert_sentiment(label):
    if label in positive:
        return "Positive"
    elif label in negative:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Sentiment"].apply(convert_sentiment)
# print(df.Sentiment.value_counts().keys())

# print(df.isnull().sum())
# print(df.Sentiment.value_counts().keys())

X = df.Text
y = df.Sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")

X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)

models = {
    "LogReg":       LogisticRegression(max_iter=1000),
    "NaiveBayes":   MultinomialNB(),
    "SVM":          LinearSVC(C=2, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

model_evaluation = {"Model Name": [], "Accuracy": [], "Precision":[], "Recall": [], "F1 Score": []}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    model_evaluation["Model Name"].append(name)
    model_evaluation["Accuracy"].append(accuracy)
    model_evaluation["Precision"].append(precision)
    model_evaluation["Recall"].append(recall)
    model_evaluation["F1 Score"].append(f1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))

    labels = ["Negative", "Neutral", "Positive"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"reports/confusion_matrix_{name}.png")
    plt.close()

model_evaluation = pd.DataFrame(model_evaluation)
model_evaluation.to_csv("reports/model_evaluation.csv")

model_evaluation.plot(x="Model Name", y=["Accuracy","F1 Score"], kind="bar")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.savefig("reports/model_comparison.png")
plt.close()

best_model_name = model_evaluation.loc[model_evaluation["F1 Score"].idxmax(), "Model Name"]
best_model = models[best_model_name]

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
labels = ["Negative", "Neutral", "Positive"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"Best Model Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(f"reports/confusion_matrix_best_model.png")
plt.close()

cm_normalized = confusion_matrix(y_test, y_pred_best, normalize='true')
plt.figure(figsize=(6,5))
labels = ["Negative", "Neutral", "Positive"]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"Best Model Confusion Matrix (Normalized) - {best_model_name}")
plt.tight_layout()
plt.savefig(f"reports/confusion_matrix_best_model_normalized.png")
plt.close()

report = classification_report(y_test, y_pred_best)
with open(f"reports/classification_report_best_model.txt", "w") as f:
    f.write(f"Model: {best_model_name}\n")
    f.write(report)

joblib.dump(best_model, f"models/model.joblib")
joblib.dump(tfidf_vectorizer, f"models/vectorizer.joblib")
print(f"{best_model_name} model saved successfully!")