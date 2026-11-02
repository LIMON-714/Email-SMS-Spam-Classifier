
import pickle
import string
from pathlib import Path

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


# ---------------- LOAD RESOURCES ----------------

@st.cache_resource
def load_resources():

    for resource in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)

    model_path = Path(__file__).with_name("model.pkl")
    vectorizer_path = Path(__file__).with_name("vectorizer.pkl")

    if not model_path.exists():
        raise FileNotFoundError("model.pkl file not found!")

    if not vectorizer_path.exists():
        raise FileNotFoundError("vectorizer.pkl file not found!")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf


# ---------------- TEXT PREPROCESSING ----------------

ps = PorterStemmer()


@st.cache_resource
def load_stopwords():
    return set(stopwords.words("english"))


# ---------------- TRANSFORM TEXT ----------------

def transform_text(text):

    text = str(text).lower()

    tokens = nltk.word_tokenize(text)

    stop_words = load_stopwords()

    cleaned = []

    for token in tokens:

        if (
            token.isalnum()
            and token not in stop_words
            and token not in string.punctuation
        ):

            cleaned.append(ps.stem(token))

    return " ".join(cleaned)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #777;
    margin-bottom: 1.5rem;
}

.result {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 700;
}

.spam {
    background: #ffe7e7;
    color: #b42318;
    border: 1px solid #f5b8b8;
}

.ham {
    background: #e5f7eb;
    color: #16703a;
    border: 1px solid #a9dfba;
}

</style>

<div class="main-title">
    📩 Email/SMS Spam Classifier
</div>

<div class="sub-title">
    Enter a message below to check whether it is Spam or Not Spam.
</div>

""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

try:

    model, tfidf = load_resources()

except Exception as e:

    st.error(f"Model Loading Error: {e}")

    st.stop()


# ---------------- SESSION STATE ----------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "processed_message" not in st.session_state:
    st.session_state.processed_message = ""

if "spam_probability" not in st.session_state:
    st.session_state.spam_probability = None


# ---------------- CLEAR FUNCTION ----------------

def clear_message():

    st.session_state.message_input = ""

    st.session_state.prediction_result = None

    st.session_state.processed_message = ""

    st.session_state.spam_probability = None


# ---------------- MESSAGE INPUT ----------------

message = st.text_area(
    "Enter your message",
    height=150,
    placeholder="Type or paste your SMS / email here...",
    key="message_input"
)


# ---------------- BUTTONS ----------------

col1, col2 = st.columns(2)

with col1:

    predict_clicked = st.button(
        "🔍 Predict",
        type="primary",
        use_container_width=True
    )

with col2:

    st.button(
        "🗑️ Clear",
        on_click=clear_message,
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if predict_clicked:

    if not message.strip():

        st.warning("Please enter a message first.")

        st.session_state.prediction_result = None
        st.session_state.spam_probability = None

    else:

        try:

            # Text preprocessing
            transformed = transform_text(message)

            # TF-IDF Transformation
            # Convert Sparse Matrix to Dense Array
            vector_input = tfidf.transform(
                [transformed]
            ).toarray()

            # Model Prediction
            prediction = int(
                model.predict(vector_input)[0]
            )

            st.session_state.prediction_result = prediction

            st.session_state.processed_message = transformed

            # ---------------- SPAM PROBABILITY ----------------

            spam_probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    vector_input
                )[0]

                classes = list(model.classes_)

                if 1 in classes:

                    spam_index = classes.index(1)

                    spam_probability = float(
                        probabilities[spam_index]
                    )

            st.session_state.spam_probability = spam_probability

        except Exception as e:

            st.error(f"Prediction Error: {e}")

            st.session_state.prediction_result = None

            st.session_state.spam_probability = None


# ---------------- DISPLAY RESULT ----------------

if st.session_state.prediction_result is not None:

    st.divider()

    prediction = st.session_state.prediction_result

    if prediction == 1:

        st.markdown(
            '<div class="result spam">⚠️ SPAM MESSAGE</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result ham">✅ NOT SPAM</div>',
            unsafe_allow_html=True
        )

    # Spam Probability

    if st.session_state.spam_probability is not None:

        probability = st.session_state.spam_probability

        st.caption(
            f"Spam Probability: {probability * 100:.2f}%"
        )

        st.progress(probability)

    # Processed Message

    with st.expander("View Processed Message"):

        st.write(st.session_state.processed_message)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Machine Learning Project • TF-IDF + Voting Classifier"
)
