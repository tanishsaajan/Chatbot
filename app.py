
from flask import Flask, render_template, request

app = Flask(__name__)
import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
# Creating ChatBot Instance

 # Training with Personal Ques & Ans 
intents = json.loads(open('training_data/intents.json').read())
intents = json.loads(open('training_data/intents.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot_model.model')
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    while True:
        message=input("")
        ints=predict_class(message)
        res=get_response(ints,intents)
        return res
def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag']  == tag:
                result = random.choice(i['responses'])
                break
        return result
def predict_class(sentence):
        p = bag_of_words(sentence)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list
def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)
if __name__ == "__main__":
    app.run() 