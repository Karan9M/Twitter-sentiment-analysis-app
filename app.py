import streamlit as st
import pickle
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# -------------------- CACHE RESOURCES --------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# -------------------- SENTIMENT FUNCTION --------------------
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# -------------------- CARD UI --------------------
def create_card(tweet_text, sentiment):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    emoji = "😊" if sentiment == "Positive" else "😞"
    card_html = f"""
    <div style="
        background-color: {color};
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    ">
        <h4 style="color: white; margin: 0;">{emoji} {sentiment} Sentiment</h4>
        <p style="color: white; font-size: 16px; margin-top: 8px;">{tweet_text}</p>
    </div>
    """
    return card_html

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🔎", layout="centered")

    st.markdown(
        "<h1 style='text-align: center; color: #1DA1F2;'>🔎 Sentiment Analyzer</h1>",
        unsafe_allow_html=True
    )
    st.write("Analyze the **sentiment** of any text instantly! 🚀")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    # Text input box with highlight
    text_input = st.text_area("✍️ Enter your text below", placeholder="Type something like: I love using Streamlit!")

    if st.button("Analyze Sentiment"):
        if text_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔍 Analyzing sentiment... Please wait"):
                time.sleep(1.5)  # fake delay for better UX
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)

            # Display result in styled card
            card_html = create_card(text_input, sentiment)
            st.markdown(card_html, unsafe_allow_html=True)

            # Extra feedback message
            if sentiment == "Positive":
                st.success("✅ This text has a positive vibe! Keep spreading good energy ✨")
            else:
                st.error("❌ This text seems negative. Stay strong 💪")

if __name__ == "__main__":
    main()
