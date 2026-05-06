## 🎬 Movie Review Sentiment Analysis (Deep Learning + Streamlit)
# 📌 Project Overview

This project is an end-to-end Sentiment Analysis application that classifies movie reviews as Positive 😊 or Negative 😞 using a Deep Learning model.

It uses a Bidirectional GRU (Gated Recurrent Unit) model trained on the IMDB dataset and is deployed as an interactive web app using Streamlit.

# 🚀 Live Demo

https://yourname.streamlit.app

# Tech Stack
1. Python
2. TensorFlow / Keras
3. Natural Language Processing (NLP)
4. Streamlit
5. BeautifulSoup (Text Cleaning)
6. Scikit-learn
7. Joblib

# How It Works
1. Text Preprocessing
   - Remove URLs, mentions, HTML tags
   - Convert to lowercase
   - Clean unwanted characters
2. Tokenization
   - Convert text → sequences using Tokenizer
   - Handle unknown words using <OOV>
3. Padding
   - Ensure fixed input length using pad_sequences
4. Model Architecture
   - Embedding Layer
   - Bidirectional GRU Layer
   - Dense Output Layer (Sigmoid)
8. Prediction
   - Output probability
   - Classify:
     > 0.5 → Positive
     < 0.5 → Negative
     
# Model Details
- Model: Bidirectional GRU
- Dataset: IMDB Movie Reviews (~50,000 samples)
- Vocabulary Size: 10,000
- Max Sequence Length: 200
- Loss Function: Binary Crossentropy
- Optimizer: Adam

**Hope you like it**
