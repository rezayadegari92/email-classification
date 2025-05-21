import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import joblib
import re

class EmailClassifier:
    def __init__(self):
        self.pipeline = None
        self.categories = [
            'Customer Support / Complaints',
            'Sales Inquiry / Pricing Request',
            'Partnership / Collaboration Proposal',
            'Spam / Advertisements'
        ]
        
    def preprocess_text(self, text):
        """Preprocess the text by removing punctuation, converting to lowercase, and removing stopwords."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare the data by combining subject and body, and preprocessing the text."""
        # Combine subject and body
        df['text'] = df['subject'] + ' ' + df['body']
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        return df
    
    def train(self, df):
        """Train the model using the provided dataset."""
        # Prepare the data
        df = self.prepare_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'],
            df['label'],
            test_size=0.2,
            random_state=42
        )
        
        # Create and train the pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', SVC(kernel='linear', probability=True))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        return self.pipeline
    
    def predict(self, subject, body):
        """Predict the category of an email."""
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")
        
        # Combine and preprocess the text
        text = self.preprocess_text(subject + ' ' + body)
        
        # Get prediction and probability
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = max(probabilities)
        
        return {
            'category': prediction,
            'confidence': float(confidence)
        }
    
    def save_model(self, path='model.joblib'):
        """Save the trained model to a file."""
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.pipeline, path)
    
    def load_model(self, path='model.joblib'):
        """Load a trained model from a file."""
        self.pipeline = joblib.load(path) 