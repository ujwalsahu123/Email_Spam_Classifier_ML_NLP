import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.set_page_config(page_title="Inbox Shield", page_icon="📨", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6f8fb 0%, #eef3f8 100%);
    }
    .hero-card {
        background: white;
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 18px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 0.98rem;
        line-height: 1.5;
    }
    .result-box {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
        font-weight: 700;
        text-align: center;
    }
    .spam-box {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .ham-box {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('model/vectorizer.pkl','rb'))
model = pickle.load(open('model/model.pkl','rb'))

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Inbox Shield</div>
        <div class="hero-subtitle">Paste a message below and check whether it looks like spam or a normal message.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown('<div class="result-box spam-box">Spam</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box ham-box">Not Spam</div>', unsafe_allow_html=True)
