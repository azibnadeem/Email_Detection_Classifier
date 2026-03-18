"""
Script to train and save the email spam detection model.
Run this once to generate vectorizer.pkl and model.pkl files.
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tools
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    """Preprocess email text"""
    text = text.lower()
    text = word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in STOPWORDS and i not in string.punctuation]
    return " ".join(y)

# Check if you have a dataset (CSV file with 'text' and 'label' columns)
try:
    # Adjust the filename and columns based on your actual dataset
    df = pd.read_csv('spam_dataset.csv')  # or whatever your dataset filename is
    print(f"Dataset loaded: {len(df)} emails")
    
    # Preprocess the text
    df['transformed_text'] = df['text'].apply(transform_text)
    
    # Train TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text'])
    y = df['label']
    
    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)
    
    # Save the models
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))
    
    print("✓ Models trained and saved successfully!")
    print(f"✓ Vectorizer saved to vectorizer.pkl")
    print(f"✓ Model saved to model.pkl")
    
except FileNotFoundError:
    print("ERROR: Dataset file not found!")
    print("Please provide a CSV file named 'spam_dataset.csv' with 'text' and 'label' columns")
    print("\nIf you don't have a dataset, you can:")
    print("1. Download a spam/ham dataset from Kaggle")
    print("2. Create a simple CSV with columns: 'text', 'label' (0=ham, 1=spam)")
