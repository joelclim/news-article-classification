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


# Load traditional model and vecotorizer
tdl_model = joblib.load('models/svm_model.pkl')
tdl_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load the deep learning model, tokenizer, and label encoder
dpl_model = load_model("models/cnn_model.keras")

with open("models/cnn_tokenizer.pkl", "rb") as handle:
    dpl_tokenizer = pickle.load(handle)

with open("models/cnn_label_encoder.pkl", "rb") as handle:
    dpl_label_encoder = pickle.load(handle)

pretrained_summarizer = pipeline("summarization")


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

    normalized_article = clean_stem_text(article_text)
    vectorized_article = tdl_vectorizer.transform([normalized_article]).toarray()
    tdl_category = tdl_model.predict(vectorized_article)[0]

    max_length = 200
    sequence = dpl_tokenizer.texts_to_sequences([article_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    dpl_prediction = dpl_model.predict(padded_sequence)
    category_index = dpl_prediction.argmax(axis=1)[0]
    dpl_category = dpl_label_encoder.inverse_transform([category_index])[0]
    
    return {
        "tdl_category": tdl_category,
        "dpl_category": dpl_category,
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
            st.write(f"**Category (SVM):** {result['tdl_category']}")
            st.write(f"**Category (Deep Learning):** {result['dpl_category']}")
            st.write(f"**Summary:** {result['summary']}")
        else:
            st.error("Please paste a news article before clicking the button.")

# Run the app
if __name__ == "__main__":
    
    main()
