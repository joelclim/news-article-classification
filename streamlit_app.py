import streamlit as st

import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import pipeline



@st.cache_resource
def load_traditional_learning_model():
    nltk.download('stopwords')
    model = joblib.load('models/svm_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    return model, vectorizer


@st.cache_resource
def load_deep_learning_model():
    model = load_model("models/cnn_model.keras")
    
    with open("models/cnn_tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    with open("models/cnn_label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)

    return model, tokenizer, label_encoder


@st.cache_resource()
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


load_traditional_learning_model()
load_deep_learning_model()
load_summarizer()


def stop_words():
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    return all_stopwords


def clean_stem_text(text):
    # replace any non-alphabet characters by a space
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    
    # replace uppercase characters to lowercase characters
    cleaned_text = cleaned_text.lower()
    
    # split text into words
    tokens = cleaned_text.split()
    
    # stem each words of each article text
    ps = PorterStemmer()
    all_stopwords = stop_words()
    stemmed_text = [ps.stem(word) for word in tokens
                  if not word in set(all_stopwords)]
    # join the words together to become a single text separated by a space
    stemmed_text = ' '.join(stemmed_text)
    
    return stemmed_text


def classify(text):
    model, vectorizer = load_traditional_learning_model()
    
    normalized_article = clean_stem_text(text)
    vectorized_article = vectorizer.transform([normalized_article]).toarray()
    category = model.predict(vectorized_article)[0]

    return category


def classify_dl(text):
    model, tokenizer, label_encoder = load_deep_learning_model()
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post')
    prediction = model.predict(padded_sequence)
    category_index = prediction.argmax(axis=1)[0]
    category = label_encoder.inverse_transform([category_index])[0]

    return category


def classify_article(article_text):
    """
    Process the news article to categorize and summarize it.
    
    Parameters:
    article_text (str): The news article text.
    
    Returns:
    dict: A dictionary containing the category and summary.
    """
    return {
        "category": classify(article_text),
        "category_dl": classify_dl(article_text),
    }


def summarize_article(article_text):
    summarizer = load_summarizer()
    summary = summarizer(article_text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# Streamlit app
def main():
    st.title('🗞️ News Article Classifier')
    st.write("Paste a news article below, and the app will categorize and summarize it.")

    # Text input area for the news article
    article_text = st.text_area("Paste your news article here:", height=300)

    classification_results = st.empty()
    classification_dl_results = st.empty()
    summarization_results = st.empty()

    # Buttons for actions
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        classify_button = st.button("Classify Article")
    with col2:
        summarize_button = st.button("Summarize Article")
    with col3:
        clear_button = st.button("Clear Results")

    if classify_button:
        if article_text.strip():
            with st.spinner("Classifying the article..."):
                classification = classify_article(article_text)
                classification_results.subheader("Classification Result using Support Vector Machine")
                classification_results.write(classification['category'])
                classification_dl_results.subheader("Classification Result using Deep Learning")
                classification_dl_results.write(classification['category_dl'])
        else:
            st.error("Please paste a news article to classify.")

    # Handle summarization
    if summarize_button:
        if article_text.strip():
            with st.spinner("Summarizing the article..."):
                summary = summarize_article(article_text)
                summarization_results.subheader("Summary")
                summarization_results.write(summary)
        else:
            st.error("Please paste a news article to summarize.")

    # Clear results
    if clear_button:
        classification_results.empty()
        classification_dl_results.empty()
        summarization_results.empty()
        st.success("Results cleared.")


# Run the app
if __name__ == "__main__":
    main()
