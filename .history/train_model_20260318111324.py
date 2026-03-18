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
import zipfile

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize tools
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    """Preprocess email text"""
    text = str(text).lower()
    text = word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in STOPWORDS and i not in string.punctuation]
    return " ".join(y)

# Create minimal fallback training data
FALLBACK_DATA = {
    'v1': ['ham'] * 20 + ['spam'] * 20,
    'v2': [
        # Ham messages
        "Hey, how are you doing today?",
        "Let's meet for coffee tomorrow at 3pm",
        "Thanks for the help yesterday",
        "Can you send me the report?",
        "Happy birthday! Hope you have a great day",
        "Don't forget about the meeting at 2",
        "The project is looking good",
        "I'll call you later tonight",
        "Great job on the presentation",
        "See you at the gym tomorrow",
        "Remember to bring your ID",
        "Book your tickets now",
        "Let me know if you need anything",
        "Good morning, have a nice day",
        "The weather is beautiful today",
        "Congratulations on your promotion",
        "Can we reschedule?",
        "Thanks for everything",
        "Looking forward to meeting you",
        "Take care and stay safe",
        # Spam messages
        "CONGRATULATIONS YOU HAVE WON FREE MONEY!!!",
        "Click here to claim your prize now",
        "You have been selected for a special offer",
        "URGENT: Your account needs to be verified",
        "Earn money fast with this amazing opportunity",
        "Limited time offer - call NOW!",
        "You are a winner! Claim your reward today",
        "Act now before this offer expires",
        "FREE MONEY just for you",
        "Increase your income with work from home",
        "LAST CHANCE to get amazing deals",
        "You have been specially chosen",
        "Do not miss this incredible opportunity",
        "CLICK HERE FOR EXCLUSIVE OFFER",
        "Your credit has been approved for 10000",
        "You qualify for FREE CASH",
        "This is a limited time offer",
        "Don't let this opportunity pass by",
        "ASAP: Confirm your account details",
        "Reply now to claim your winnings"
    ]
}

def load_from_kaggle():
    """Try to load from local spam.csv file"""
    try:
        if os.path.exists('spam.csv'):
            print("Loading dataset from spam.csv...")
            df = pd.read_csv('spam.csv', encoding='latin1')
            if 'v1' in df.columns and 'v2' in df.columns:
                return df
            elif 'target' in df.columns and 'text' in df.columns:
                df.rename(columns={'target': 'v1', 'text': 'v2'}, inplace=True)
                return df
    except Exception as e:
        pass
    return None

def load_from_uci():
    """Try to download from UCI machine learning repository"""
    try:
        print("Attempting to download UCI spam dataset...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        response = urllib.request.urlopen(url, timeout=10)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        
        # Extract and read the data
        with zip_file.open('SMSSpamCollection') as f:
            data = f.read().decode('utf-8').split('\n')
            rows = [line.split('\t', 1) for line in data if line.strip()]
            df = pd.DataFrame(rows, columns=['v1', 'v2'])
            print(f"Successfully downloaded {len(df)} messages from UCI")
            return df
    except Exception as e:
        print(f"Could not download from UCI: {e}")
    return None

def load_dataset():
    """Load dataset from various sources"""
    # Try local file first
    df = load_from_kaggle()
    if df is not None:
        return df
    
    # Try UCI download
    df = load_from_uci()
    if df is not None:
        return df
    
    # Use fallback data
    print("Using fallback training data for demonstration...")
    df = pd.DataFrame(FALLBACK_DATA)
    return df

try:
    # Load dataset
    df = load_dataset()
    
    # Ensure proper column names
    if 'v1' not in df.columns:
        df.rename(columns={df.columns[0]: 'v1', df.columns[1]: 'v2'}, inplace=True)
    
    # Create label column
    df['label'] = (df['v1'].str.lower() == 'spam').astype(int)
    df['text'] = df['v2']
    
    print(f"Dataset loaded: {len(df)} messages")
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
    print("\nNOTE: For better accuracy, obtain and use the full UCI SMS Spam Collection dataset:")
    print("Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("Place spam.csv in this directory and run this script again.")
    
except Exception as e:
    print(f"\nERROR during training: {e}")
    import traceback
    traceback.print_exc()
    print("\nAlternative solutions:")
    print("1. Manually download the dataset from Kaggle")
    print("2. Place spam.csv in this directory")
    print("3. Run this script again")
