"""
Script to train and save the email spam detection model.
Downloads public spam dataset and trains the MultinomialNB model.
Run this once to generate vectorizer.pkl and model.pkl files.
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import os
import urllib.request
import io

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize tools
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    """Preprocess email text"""
    text = text.lower()
    text = word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in STOPWORDS and i not in string.punctuation]
    return " ".join(y)

# Function to download dataset from URL
def download_dataset(url, local_file):
    """Download dataset from URL if it doesn't exist locally"""
    if os.path.exists(local_file):
        print(f"Using existing dataset: {local_file}")
        return local_file
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, local_file)
        print(f"Dataset downloaded successfully to {local_file}")
        return local_file
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# Try to load dataset
dataset_file = 'spam.csv'
dataset_url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/spam.csv'

try:
    # Check if spam.csv exists locally
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found locally. Attempting to download...")
        status = download_dataset(dataset_url, dataset_file)
        if status is None:
            raise FileNotFoundError("Could not download dataset")
    
    # Load the dataset
    print(f"Loading dataset from {dataset_file}...")
    df = pd.read_csv(dataset_file, encoding='latin1')
    
    # Rename columns if needed (for the Kaggle spam dataset)
    if 'v1' in df.columns and 'v2' in df.columns:
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    
    # Convert labels to binary (0 = ham/not spam, 1 = spam)
    if 'target' in df.columns:
        df['label'] = (df['target'] == 'spam').astype(int)
    elif 'label' not in df.columns:
        raise ValueError("Dataset must have 'target' or 'label' column")
    
    print(f"Dataset loaded: {len(df)} emails")
    print(f"Spam count: {(df['label'] == 1).sum()}")
    print(f"Ham count: {(df['label'] == 0).sum()}")
    
    # Preprocess the text
    print("Preprocessing text...")
    df['transformed_text'] = df['text'].apply(transform_text)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['transformed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Train TF-IDF Vectorizer
    print("Training TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Train Naive Bayes model
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate on test set
    X_test_tfidf = tfidf.transform(X_test)
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save the models
    print("\nSaving models...")
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))
    
    print("\n✓ SUCCESS! Models trained and saved!")
    print(f"✓ Vectorizer saved to: vectorizer.pkl")
    print(f"✓ Model saved to: model.pkl")
    print(f"✓ Test Accuracy: {accuracy:.2%}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nTo fix this, you have two options:")
    print("1. Create a 'spam.csv' file with columns 'v1' (label) and 'v2' (text)")
    print("2. Download the spam dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("\nIf you have an offline dataset already, place the CSV file in this directory.")
