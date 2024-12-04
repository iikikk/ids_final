from flask import Flask, request, jsonify
import logging
from gensim.models import Word2Vec
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
import re

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained models
WORD2VEC_MODEL_PATH = 'word2vec_twitter.model'
RANDOMFOREST_MODEL_PATH = 'randomforest_bully_predictor.pkl'

try:
    logger.info("Loading Word2Vec model...")
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    logger.info("Word2Vec model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Word2Vec model: {e}")

try:
    logger.info("Loading RandomForest model...")
    randomforest_model = joblib.load(RANDOMFOREST_MODEL_PATH)
    logger.info("RandomForest model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load RandomForest model: {e}")

# Helper function to clean text
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    tweet = re.sub(r'\bRT\b', '', tweet)
    return tweet.lower().strip()

# Helper function to generate embeddings
def get_embedding(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t in word2vec_model.wv.key_to_index]
    if tokens:
        return np.mean([word2vec_model.wv[token] for token in tokens], axis=0)
    return np.zeros(100)  # Assuming vector size is 100

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        logger.info(f"Received text for prediction: {text}")

        # Clean and process text
        cleaned_text = clean_tweet(text)
        embedding = get_embedding(cleaned_text).reshape(1, -1)

        # Make prediction
        prediction = randomforest_model.predict(embedding)[0]
        probabilities = randomforest_model.predict_proba(embedding).tolist()

        logger.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
        return jsonify({'prediction': int(prediction), 'probabilities': probabilities})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
