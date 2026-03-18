import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# 1. Setup NLTK (Essential for Deployment)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    # Remove non-alphanumeric
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# 2. Load Models Safely
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()

# 3. UI Implementation
st.title("📧 Email Classifier")
html_temp = """
<div style="background-color:tomato;padding:10px;border-radius:10px">
<h2 style="color:white;text-align:center;">Streamlit Email Spam Detector </h2>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message you want to check:", height=150)

if st.button('Analyze Message'):
    if input_sms.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Process
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict
        prob = model.predict_proba(vector_input)[0]
        result = model.predict(vector_input)[0]

        # Display Result
        if result == 1:
            st.error(f"🚨 **Spam detected!** (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Not Spam.** (Confidence: {prob[0]*100:.2f}%)")

# Footer
st.markdown("---")
st.write("Copy© 2026 Adeel Munir | Made With ❤️ in Pakistan")