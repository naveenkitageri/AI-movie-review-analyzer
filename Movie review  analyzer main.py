import os
import re
import pandas as pd
import nltk 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from bs4 import BeautifulSoup
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Text Hyperparameters
 
vocab_size = 10000
max_len = 200


def load_path(path):
    df = pd.read_csv(path)
    print('Imported data...')
    return df


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


def split(df):
    X = df['clean_review']
    y = df['sentiment']

    y = y.map({'positive':1, 'negative':0})

    # Train and Test split

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    
    # Convert text to word index

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    # Convert word index to number

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Padding (add 0) with limited length

    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    return X_train, y_train, X_test, y_test, tokenizer

def build_model(X_train, X_test, y_train, y_test):

    # Bulding Model

    bidi_gru_model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        layers.Bidirectional(layers.GRU(64, dropout=0.2)),
        layers.Dense(1, activation='sigmoid')
    ])
  
    bidi_gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    bidi_gru_model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))
    return bidi_gru_model


def main():
    BASE_DIR = os.path.dirname(os.abspath(__file__))
    path = os.path.dirname(BASE_DIR, "IMDB_dataset-1.csv")
    df = load_path(path)

    df['clean_review'] = df['review'].apply(clean_text)

    X_train, y_train, X_test, y_test, tokenizer = split(df)
    
    bidi_gru_model = build_model(X_train, X_test, y_train, y_test)

    #  Model Evaluation  

    y_pred_prob = bidi_gru_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int')

    print('Confusion matrix :\n', confusion_matrix(y_test, y_pred))
    print('Classification Report :\n', classification_report(y_test, y_pred))

    print('Model Trained.')

    dump(tokenizer, tokenizer.joblib')
    print('Tokenizer saved')

    bidi_gru_model.save('bidirectional_gru_model.h5')
    print('Model saved')    

if __name__ == '__main__':
    main()
