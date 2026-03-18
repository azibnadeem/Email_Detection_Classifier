# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.stem.porter import PorterStemmer


# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords') 
# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:

#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)


# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
# # with open("vectorizer.pkl", "rb") as f:
# #     tfidf = pickle.load(f)

# # with open("model.pkl", "rb") as f:
# #     model = pickle.load(f)


# st.title("Email Classifier")
# html_temp = """
# <div style="background-color:tomato;padding:10px">
# <h2 style="color:white;text-align:center;">Streamlit Email Spam Detector ML App </h2>
# </div>
# <br>
#   """
# st.markdown(html_temp, unsafe_allow_html=True)

# input_sms = st.text_area("Enter the message")

# if st.button('Inquire'):
#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict probability
#     prob = model.predict_proba(vector_input)[0]
#     spam_prob = prob[1] * 100
#     ham_prob = prob[0] * 100


#     # Optional: display final label
#     result = model.predict(vector_input)[0]
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
    
#      # 4. Display the probabilities
#     st.write(f"Spam Probability: {spam_prob:.2f}%")
#     st.write(f"Not Spam Probability: {ham_prob:.2f}%")

# st.write("\n\n\n\n\n")

# st.write("\n" * 15)
# # Add a bold line above the footer
# st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# # Footer content
# st.write("Copy© 2026 Adeel Munir | Made With ❤️ in Pakistan")

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# 1. NLTK Downloads (Run once)
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

load_nltk()

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english')) # Loaded once for speed

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    # Remove non-alphanumeric and stopwords/punctuation
    y = [ps.stem(i) for i in text if i.isalnum() and i not in STOPWORDS and i not in string.punctuation]
    
    return " ".join(y)

# 2. Optimized Model Loading
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

tfidf, model = load_models()

# 3. UI/UX
st.title("Email Classifier")
html_temp = """
<div style="background-color:tomato;padding:10px;border-radius:10px">
<h2 style="color:white;text-align:center;">Streamlit Email Spam Detector ML App </h2>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message", height=150)

if st.button('Inquire'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to analyze.")
    elif tfidf is None or model is None:
        st.error("Models are not loaded. Please check that vectorizer.pkl and model.pkl exist and are valid.")
    else:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            prob = model.predict_proba(vector_input)[0]
            result = model.predict(vector_input)[0]

            # 4. Display results
            if result == 1:
                st.error("This is SPAM [Alert]")
            else:
                st.success("This is NOT SPAM (Ham) [Safe]")

            # Probabilities in a nice layout
            col1, col2 = st.columns(2)
            col1.metric("Spam Probability", f"{prob[1]*100:.2f}%")
            col2.metric("Not Spam Probability", f"{prob[0]*100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("<br><br><hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
st.info("Copy© 2026 Adeel Munir | Made With Love in Pakistan")