import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# --- 1. NLTK Resource Setup ---
# This ensures the resources are available in the Streamlit Cloud environment
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res != 'stopwords' else f'corpora/{res}')
        except LookupError:
            nltk.download(res)

download_nltk_resources()
ps = PorterStemmer()

# --- 2. Text Preprocessing Function ---
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

# --- 3. Secure Model Loading ---
@st.cache_resource 
def load_models():
    # Get the absolute path to the directory this script is in
    base_path = os.path.dirname(__file__)
    
    tfidf_path = os.path.join(base_path, 'vectorizer.pkl')
    model_path = os.path.join(base_path, 'model.pkl')
    
    # Check if files exist before trying to open them
    if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
        st.error(f"Error: Required files not found in {base_path}. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the repository.")
        st.stop()

    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return tfidf, model

# Load the models
tfidf, model = load_models()

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

st.title("📧 Email Classifier")

html_temp = """
<div style="background-color:tomato;padding:15px;border-radius:10px;margin-bottom:20px">
<h2 style="color:white;text-align:center;margin:0;">Streamlit Email Spam Detector</h2>
<p style="color:white;text-align:center;margin:5px 0 0 0;">AI-powered classification by Adeel Munir</p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message you want to check:", height=150, placeholder="Type or paste your email content here...")

if st.button('Analyze Message', use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner('Processing...'):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. Predict
            result = model.predict(vector_input)[0]
            prob = model.predict_proba(vector_input)[0]

            # 4. Display Results
            st.markdown("---")
            if result == 1:
                st.error(f"### 🚨 Result: **Spam Detected**")
                st.progress(prob[1]) # Progress bar for visual confidence
                st.write(f"Confidence Level: **{prob[1]*100:.2f}%**")
            else:
                st.success(f"### ✅ Result: **Not Spam**")
                st.progress(prob[0])
                st.write(f"Confidence Level: **{prob[0]*100:.2f}%**")

# --- 5. Footer ---
st.write("\n" * 5)
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)
st.info("Copy© 2026 Adeel Munir | Made With ❤️ in Pakistan")