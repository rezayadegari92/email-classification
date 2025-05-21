import pandas as pd
from model import EmailClassifier
import nltk

def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('email_dataset.csv')
    
    # Initialize and train the model
    print("Training model...")
    classifier = EmailClassifier()
    classifier.train(df)
    
    # Save the model
    print("Saving model...")
    classifier.save_model()
    print("Model saved as 'model.joblib'")

if __name__ == "__main__":
    main() 