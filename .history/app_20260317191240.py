import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted

# --- 1. NLTK Resource Setup ---
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

# --- 2. Text Preprocessing ---
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

# --- 3. Secure & Validated Model Loading ---
@st.cache_resource 
def load_models():
    base_path = os.path.dirname(__file__)
    tfidf_path = os.path.join(base_path, 'vectorizer.pkl')
    model_path = os.path.join(base_path, 'model.pkl')
    
    # Check if files exist
    if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
        st.error(f"❌ Files not found! Ensure 'vectorizer.pkl' and 'model.pkl' are in your GitHub root.")
        st.stop()

    try:
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Validation: Check if the model is actually trained
        check_is_fitted(model)
        
        return tfidf, model
    except EOFError:
        st.error("❌ **EOFError:** The model files are empty or corrupted. Re-upload them to GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error Loading Model:** {e}")
        st.stop()

# Initialize models
tfidf, model = load_models()

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email Spam Classifier")

# Design Header
st.markdown("""
    <div style="background-color:tomato;padding:15px;border-radius:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;margin:0;">AI Message Detector</h2>
    <p style="color:white;text-align:center;margin:5px 0 0 0;">Developed by Adeel Munir</p>
    </div>
    """, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message below:", height=150, placeholder="Paste email or SMS content here...")

if st.button('Analyze Message', use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
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
            st.error("### 🚨 Result: **SPAM**")
            st.write(f"Confidence Level: **{prob[1]*100:.2f}%**")
            st.progress(prob[1])
        else:
            st.success("### ✅ Result: **NOT SPAM (HAM)**")
            st.write(f"Confidence Level: **{prob[0]*100:.2f}%**")
            st.progress(prob[0])

# --- 5. Footer ---
st.write("\n" * 3)
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Copy© 2026 Adeel Munir | Made With ❤️ in Pakistan</p>", unsafe_allow_html=True)