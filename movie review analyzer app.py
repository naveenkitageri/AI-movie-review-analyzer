import os
import re
import streamlit as st
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from joblib import load

st.set_page_config(page_title='Movie sentiment analysis', layout='centered')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bidirectional_gru_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.joblib")
MAX_LENGTH = 200

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = load(TOKENIZER_PATH)

# clean text function 

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('not good', 'bad')
    text = text.replace('not great', 'bad')
    text = text.replace('not happy', 'sad')
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    return text

# UI

st.title('AI Movie Review Analyzer')

user_input = st.text_area('Enter your movie review :')

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please entre a review')
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

        # Prediction
        pred_prob = model.predict(padded)[0][0]

        if pred_prob > 0.5:
            st.success(f'Positive Review 😊 {pred_prob:.2f}')
        else:
            st.error(f'Negative Review 😞 {pred_prob:.2f}')