import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes  import MultinomialNB
from sklearn.svm          import LinearSVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import joblib

from text_preprocessing import preprocess, preprocess_label

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

tfidf_vectorizer = TfidfVectorizer()

X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "SVM": LinearSVC(C=2),
    "DecisionTree": DecisionTreeClassifier(max_depth=10),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

model_evaluation = {"Model Name": [], "Accuracy": [], "Recall": [], "Precision":[], "F1 Score": []}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    model_evaluation["Model Name"].append(name)
    model_evaluation["Accuracy"].append(accuracy)
    model_evaluation["Recall"].append(recall)
    model_evaluation["Precision"].append(precision)
    model_evaluation["F1 Score"].append(f1)

model_evaluation = pd.DataFrame(model_evaluation)
model_evaluation.to_csv("data/model_evaluation.csv")


best_model = model_evaluation.loc[model_evaluation["F1 Score"].idxmax(), "Model Name"]
joblib.dump(models["SVM"], f"models/model.joblib")
joblib.dump(tfidf_vectorizer, f"models/vectorizer.joblib")
print(f"{best_model} model saved successfully!")