import streamlit as st

import joblib
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import pipeline


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


@st.cache_resource
def get_traditional_learning_model():
    model = joblib.load('models/svm_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    return model, vectorizer


@st.cache_resource
def get_deep_learning_model():
    model = load_model("models/cnn_model.keras")
    
    with open("models/cnn_tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    with open("models/cnn_label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)

    return model, tokenizer, label_encoder


@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


def classify(text):
    model, vectorizer = get_traditional_learning_model()
    
    normalized_article = clean_stem_text(text)
    vectorized_article = vectorizer.transform([normalized_article]).toarray()
    category = model.predict(vectorized_article)[0]

    return category


def classify_dl(text):
    model, tokenizer, label_encoder = get_deep_learning_model()
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post')
    prediction = model.predict(padded_sequence)
    category_index = prediction.argmax(axis=1)[0]
    category = label_encoder.inverse_transform([category_index])[0]

    return category


def summarize(text):
     return get_summarizer().summarizer(text, max_length=150, min_length=30, do_sample=False)


# Placeholder function for categorization and summarization
def process_article(article_text):
    """
    Process the news article to categorize and summarize it.
    
    Parameters:
    article_text (str): The news article text.
    
    Returns:
    dict: A dictionary containing the category and summary.
    """

    # Mock data to simulate behavior
    # mock_output = {
    #     "category": "Technology",
    #     "summary": "This article discusses the advancements in AI and its applications in various industries."
    # }
    # return mock_output

    return {
        "category": classify(article_text),
        "category_dl": classify_dl(article_text),
        "summary": "This article discusses the advancements in AI and its applications in various industries."
    }


# Streamlit app
def main():
    st.title('🗞️ News Article Classifier')
    st.write("Paste a news article below, and the app will categorize and summarize it.")

    # Text input area for the news article
    article_text = st.text_area("Paste your news article here:", height=300)

    # Button to process the article
    if st.button("Categorize and Summarize"):
        if article_text.strip():
            # Process the article using the placeholder function
            result = process_article(article_text)
            # Display the results
            st.subheader("Results")
            st.write(f"**Category (SVM):** {result['category']}")
            st.write(f"**Category (Deep Learning):** {result['category_dl']}")
            st.write(f"**Summary:** {result['summary']}")
        else:
            st.error("Please paste a news article before clicking the button.")


# Run the app
if __name__ == "__main__":
    
    main()
