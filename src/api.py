from fastapi import FastAPI
import joblib

from .text_preprocessing import preprocess

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API"}

vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/model.joblib")

@app.post("/predict")
def classify(input_text):
    text = preprocess(input_text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return {"text": input_text, "prediction": prediction}