import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Example usage:
#candidate_translation = "When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Pant Answer 3. Slip. In practically every home and workplace, there are electrical appliances and machinery. Although they are common and practical, they can also be quite dangerous. Thousands of people suffer amazed each year."
#reference_translations = ["When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Trip Answer 3. Slip. Electrical appliances and machinery are found in virtually every home and workplace. While they are common and convenient, they can also be quite dangerous. Thousands of people are shocked every year."]
#reference_text = "When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Trip Answer 3. Slip. Electrical appliances and machinery are found in virtually every home and workplace. While they are common and convenient, they can also be quite dangerous. Thousands of people are shocked every year."

def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity

candidate_translation = st.text_area("candidate_translation")
reference_translation = st.text_area("reference_translation")
st.button("Calculate Metrics", type="primary")

if st.button:
  reference_translations = [reference_translation]
  candidate_tokens = candidate_translation.split()
  references_tokens = [ref.split() for ref in reference_translations]
  st.write("BLEU Score:", round(corpus_bleu([references_tokens], [candidate_tokens]),2))
  st.write("Meteor Score:", round(meteor([word_tokenize(candidate_translation)],word_tokenize(reference_translation)), 2))
  st.write("Meteor Score2:", round(meteor([word_tokenize(reference_translation)],word_tokenize(candidate_translation)), 2))
  st.write("text_similarity:", text_similarity(candidate_translation, reference_translation))
