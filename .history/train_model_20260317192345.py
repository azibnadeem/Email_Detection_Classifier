import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# SAME preprocessing as app.py
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load dataset
df = pd.read_csv('spam.csv')   # ⚠️ make sure columns correct

# Example (fix column names if needed)
df = df[['v1','v2']]
df.columns = ['target','text']

# Convert labels
df['target'] = df['target'].map({'ham':0, 'spam':1})

# Apply preprocessing
df['text'] = df['text'].apply(transform_text)

# Features & labels
X = df['text']
y = df['target']

# Vectorizer
tfidf = TfidfVectorizer(max_features=3000)
X_vectorized = tfidf.fit_transform(X)

# Train model ✅
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save files ✅
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved successfully!")