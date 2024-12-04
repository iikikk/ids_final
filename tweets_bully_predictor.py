import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load and preprocess the dataset
csv_file = 'twitter_parsed_dataset.csv'
df = pd.read_csv(csv_file)

# Select relevant columns and rename them
df = df[['Text', 'oh_label']]
df.rename(columns={'Text': 'text', 'oh_label': 'labels'}, inplace=True)
df = df.dropna(subset=['labels'])

# Function to clean tweets
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

# Clean text data
df['text'] = df['text'].apply(clean_tweet)

# Function to handle class imbalance using oversampling
def oversample_data(text_column, label_column):
    sampler = RandomOverSampler(random_state=13)
    resampled_text, resampled_labels = sampler.fit_resample(text_column.values.reshape(-1, 1), label_column.values)
    return resampled_text.ravel(), resampled_labels

# Text embedding utility using Word2Vec
class TextEmbedder:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=3):
        self.model = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def train(self, texts):
        tokenized = [word_tokenize(sentence) for sentence in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size,
                              window=self.window, min_count=self.min_count, workers=4)

    def get_embedding(self, text):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if self.model and t in self.model.wv.key_to_index]
        if tokens:
            return np.mean([self.model.wv[token] for token in tokens], axis=0)
        return np.zeros(self.vector_size)

    def transform(self, texts):
        return np.array([self.get_embedding(text) for text in texts])

# Classification utility using RandomForest
class TextClassifier:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier(n_estimators=100, random_state=13)

    def train(self, features, labels):
        self.model.fit(features, labels)

    def evaluate(self, features, labels):
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        loss = log_loss(labels, probabilities)
        conf_matrix = confusion_matrix(labels, predictions)
        class_report = classification_report(labels, predictions)
        return acc, f1, loss, conf_matrix, class_report

# Main pipeline to train and evaluate the model
def word2vec_randomforest_pipeline(dataframe):
    # Handle class imbalance
    text_data, labels = oversample_data(dataframe['text'], dataframe['labels'])

    # Train Word2Vec model and generate embeddings
    embedder = TextEmbedder()
    embedder.train(text_data)
    embeddings = embedder.transform(text_data)

    # Split data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(embeddings, labels, test_size=0.2, random_state=11)

    # Train and evaluate the RandomForest model
    classifier = TextClassifier()
    classifier.train(train_X, train_y)
    acc, f1, loss, conf_matrix, class_report = classifier.evaluate(test_X, test_y)

    # Save models
    embedder.model.save('word2vec_twitter.model')
    joblib.dump(classifier.model, 'randomforest_bully_predictor.pkl')

    # Print metrics
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Log Loss: {loss}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return classifier, embedder

if __name__ == "__main__":
    word2vec_randomforest_pipeline(df)
