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

