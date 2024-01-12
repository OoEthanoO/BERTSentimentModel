import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

def encode_text_prediction(text):
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    return encoded_text

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    if user_text == '':
        return jsonify({'sentiment': 'Please enter text', 'probabilities': [0, 0, 0, 0, 0]})
    encoded_text = encode_text_prediction(user_text)
    with torch.no_grad():
        outputs = model(**encoded_text)
    predicted_class_idx = torch.argmax(outputs.logits).item()
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    predicted_sentiment = sentiment_map[predicted_class_idx]
    #return jsonify({'sentiment': predicted_sentiment})
    return jsonify({'sentiment': predicted_sentiment, 'probabilities': outputs.logits.tolist()})

@app.route('/version')
def version():
    model_summary = str(model)
    return render_template('version.html', model_summary=model_summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)