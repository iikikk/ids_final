import re
import numpy as np
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Preprocess and clean tweets
def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove mentions and hashtags
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)
    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    # Remove 'RT' (Retweet) indicator
    tweet = re.sub(r'\bRT\b', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet.strip()

# Load models
def load_models(word2vec_path, randomforest_path):
    # Load the Word2Vec model
    word2vec_model = Word2Vec.load(word2vec_path)
    # Load the RandomForest model
    randomforest_model = joblib.load(randomforest_path)
    return word2vec_model, randomforest_model

# Generate embeddings for testing
def get_embedding(text, word2vec_model, vector_size=100):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t in word2vec_model.wv.key_to_index]
    if tokens:
        return np.mean([word2vec_model.wv[token] for token in tokens], axis=0)
    return np.zeros(vector_size)

# Test the model with new data
def test_model(word2vec_model, randomforest_model, test_texts):
    # Clean the input texts
    cleaned_texts = [clean_tweet(text) for text in test_texts]
    # Generate embeddings
    embeddings = np.array([get_embedding(text, word2vec_model) for text in cleaned_texts])
    # Predict using the RandomForest model
    predictions = randomforest_model.predict(embeddings)
    probabilities = randomforest_model.predict_proba(embeddings)
    return predictions, probabilities

if __name__ == "__main__":
    # Define paths to the models
    word2vec_path = 'word2vec_twitter.model'
    randomforest_path = 'randomforest_bully_predictor.pkl'

    # Load the models
    word2vec_model, randomforest_model = load_models(word2vec_path, randomforest_path)

    # Test data
    test_texts = [
        "no no no no no no",
        "Everyone deserves kindness and understanding.",
        "now you idiots claim that people who tried to stop him from becoming a terrorist made him a terrorist islamically brain dead",
        "theres something wrong when a girl wins wayne rooney street striker"
    ]

    # Test the models
    predictions, probabilities = test_model(word2vec_model, randomforest_model, test_texts)

    # Display results
    for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
        print(f"\nTest Example {i + 1}:")
        print(f"Text: {text}")
        print(f"Predicted Label: {pred}")
        print(f"Probabilities: {prob}")
