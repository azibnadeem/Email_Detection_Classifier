# import streamlit as st
# import pickle
# import string
# import os
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer

# # --- 1. NLTK Resource Setup ---
# @st.cache_resource
# def download_nltk_resources():
#     resources = ['punkt', 'stopwords']   # ❌ removed 'punkt_tab'
#     for res in resources:
#         try:
#             nltk.data.find(f'tokenizers/{res}' if res != 'stopwords' else f'corpora/{res}')
#         except LookupError:
#             nltk.download(res)

# download_nltk_resources()

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))   # ✅ moved outside function

# # --- 2. Text Preprocessing Function ---
# def transform_text(text):
#     text = text.lower()
#     text = word_tokenize(text)

#     # Remove non-alphanumeric
#     y = [i for i in text if i.isalnum()]

#     # Remove stopwords and punctuation
#     y = [i for i in y if i not in stop_words and i not in string.punctuation]

#     # Stemming
#     y = [ps.stem(i) for i in y]

#     return " ".join(y)

# # --- 3. Load Model and Vectorizer ---
# @st.cache_resource
# def load_models():
#     base_path = os.path.dirname(__file__)

#     tfidf_path = os.path.join(base_path, 'vectorizer.pkl')
#     model_path = os.path.join(base_path, 'model.pkl')

#     if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
#         st.error("❌ model.pkl or vectorizer.pkl not found!")
#         st.stop()

#     with open(tfidf_path, 'rb') as f:
#         tfidf = pickle.load(f)

#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)

#     return tfidf, model

# tfidf, model = load_models()

# # --- 4. UI ---
# st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

# st.title("📧 Email Classifier")

# st.markdown("""
# <div style="background-color:tomato;padding:15px;border-radius:10px;margin-bottom:20px">
# <h2 style="color:white;text-align:center;margin:0;">Streamlit Email Spam Detector</h2>
# <p style="color:white;text-align:center;margin:5px 0 0 0;">AI-powered classification</p>
# </div>
# """, unsafe_allow_html=True)

# input_sms = st.text_area(
#     "Enter the message you want to check:",
#     height=150,
#     placeholder="Type or paste your email content here..."
# )

# # --- 5. Prediction ---
# if st.button('Analyze Message', use_container_width=True):

#     if not input_sms.strip():
#         st.warning("⚠️ Please enter some text.")
#     else:
#         with st.spinner('Processing...'):

#             # Preprocess
#             transformed_sms = transform_text(input_sms)

#             # Vectorize
#             vector_input = tfidf.transform([transformed_sms])

#             # Debug check (VERY IMPORTANT)
#             if not hasattr(model, "classes_"):
#                 st.error("❌ Model is NOT trained. Please retrain and upload correct model.pkl")
#                 st.stop()

#             # Predict
#             result = model.predict(vector_input)[0]
#             prob = model.predict_proba(vector_input)[0]

#             # Output
#             st.markdown("---")

#             if result == 1:
#                 st.error("🚨 Spam Detected")
#                 st.progress(float(prob[1]))
#                 st.write(f"Confidence: {prob[1]*100:.2f}%")
#             else:
#                 st.success("✅ Not Spam")
#                 st.progress(float(prob[0]))
#                 st.write(f"Confidence: {prob[0]*100:.2f}%")

# # --- Footer ---
# st.markdown("<hr>", unsafe_allow_html=True)
# st.info("© 2026 Email Spam Classifier")

import streamlit as st
import pickle

tfid = pickle.load()