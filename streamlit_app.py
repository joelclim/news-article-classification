import streamlit as st
st.set_page_config(layout="wide")

# Initialize the previous article text state to empty.
if "prev_article" not in st.session_state:
    st.session_state["prev_article"] = ""

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
def load_classical_learning_model():
    """
    Loads a (classical) machine learning model.
    
    Returns:
    model: the machine learning model.
    vectorizer: the vectorizer that represents the compute matrix for the "bag of words" model.
    """
    nltk.download('stopwords')
    model = joblib.load('models/svm_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    return model, vectorizer


@st.cache_resource
def load_deep_learning_model():
    """
    Loads a deep learning  model.
    
    Returns:
    model: the machine learning model.
    tokenizer: the tokenizer that splits the article text into input tokens to the model.
    label_encoder: the encoder/decoder of classification labels.
    """
    model = load_model("models/cnn_model.keras")
    
    with open("models/cnn_tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    with open("models/cnn_label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)

    return model, tokenizer, label_encoder


@st.cache_resource()
def load_summarizer():
    """
    Loads a summarization pipeline from HuggingFace.
    
    Returns:
    summarizer: the summarization pipeline.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")


# Loads and caches the models and summarizer immediately when the application starts
load_classical_learning_model()
load_deep_learning_model()
load_summarizer()


@st.cache_resource()
def stop_words():
    """
    Gets the list of NTLK English stopwords with modifications.
    
    Returns:
    stop_words: the NTLK English stopwords.
    """
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    return all_stopwords


def clean_stem_text(text):
    """
    Cleans and stems the article text for use in the "Bag of Words" model.
    
    Parameters:
    text (str): the news article text.
    
    Returns:
    stemmed_text: cleaned and stemmed version of the article text.
    """
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
    """
    Classifies the article text using a classical machine learning model. 
    The article is classified as either business, entertainment, politics, sport, or tech.
    
    Parameters:
    text (str): the news article text.
    
    Returns:
    category: the article's classification.
    """
    model, vectorizer = load_classical_learning_model()
    
    normalized_article = clean_stem_text(text)
    vectorized_article = vectorizer.transform([normalized_article]).toarray()
    category = model.predict(vectorized_article)[0]

    return category


def classify_dl(text):
    """
    Classifies the article text using a deep learning model. 
    The article is classified as either business, entertainment, politics, sport, or tech.
    
    Parameters:
    text (str): the news article text.
    
    Returns:
    category: the article's classification.
    """
    model, tokenizer, label_encoder = load_deep_learning_model()
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post')
    prediction = model.predict(padded_sequence)
    category_index = prediction.argmax(axis=1)[0]
    category = label_encoder.inverse_transform([category_index])[0]

    return category


def summarize_article(text):
    """
    Summarizes the article text.
    
    Parameters:
    text (str): the news article text.
    
    Returns:
    summary: the article's summary.
    """
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def sample_article():
    """
    A sample article.

    Returns:
    sample_article: a sample article.
    """    
    return '''
    The kiwifruit industry has experienced remarkable growth, with its global market value surpassing A$10 billion in 2018. Projections indicate that by 2025, global consumption will approach 6 million tonnes, expanding at an annual rate of 3.9%. Zespri, the world's largest kiwifruit marketer, manages over 30% of global supply, collaborating with more than 2,500 growers in New Zealand and 1,500 internationally. To streamline its complex operations and meet rising demand, Zespri adopted SAP S/4HANA Cloud, enhancing its supply chain management and data analysis capabilities. This digital transformation enables Zespri to efficiently deliver ripe kiwifruit year-round to consumers worldwide.
    '''


# The main method of the Streamlit app
def main():
    st.title('🗞️ News Article Classification and Summarization')
    st.markdown('The application classifies and summarizes a given article text. The article is classified as either business, entertainment, politics, sport, or tech.')
    st.markdown('''
    Two different machine learning models are used to classify the news article. The first model, a Support Vector Machine (SVM) model, is a classical machine learning model that uses the "Bag of Words" model representation of the article text.
    Other classical models considered were Naive Bayes and Random Forest, but the SVM model outperformed the former models during validation and testing.
    The second model is Deep Learning model that uses Word Embeddings as the representation of the article text and Convolutional Neural Network as the learning model.
    A [Hugging Face](https://huggingface.co/) pre-trained model is used for news summarization.
    '''.strip());
    st.markdown('This is a [SuperDataScience Community](https://community.superdatascience.com/) project.')


    # Text input area for the news article
    article_text = st.text_area("Paste your news article here:", value=sample_article().strip(), height=300)
        
    classification_header = st.empty()
    classification_results = st.empty()
    classification_dl_results = st.empty()
    summarization_header = st.empty()
    summarization_results = st.empty()

    # Check if the article text has changed
    if article_text != st.session_state["prev_article"]:
        st.session_state["prev_article"] = article_text
        classification_header.empty()
        classification_results.empty()
        classification_dl_results.empty()
        summarization_header.empty()
        summarization_results.empty()
    
    # Buttons for actions
    _, _, col1, col2, _, _ = st.columns(6)
    with col1:
        classify_button = st.button("Classify Article")
    with col2:
        summarize_button = st.button("Summarize Article")

    if classify_button:
        if article_text.strip():
            with st.spinner("Classifying the article..."):
                classification_header.subheader("Classification", divider=True)
                classification_results.markdown(f'### Using Support Vector Machine model: :blue[{classify(article_text)}]')
                classification_dl_results.markdown(f'### Using Deep Learning model (CNN): :orange[{classify_dl(article_text)}]')
        else:
            st.error("Please paste a news article to classify.")

    # Handle summarization
    if summarize_button:
        if article_text.strip():
            with st.spinner("Summarizing the article..."):
                summary = summarize_article(article_text)
                summarization_header.subheader("Summary", divider=True)
                summarization_results.write(summary)
        else:
            st.error("Please paste a news article to summarize.")


# Run the app
if __name__ == "__main__":
    main()
