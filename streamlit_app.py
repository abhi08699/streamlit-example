import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
# Example usage:
#candidate_translation = "When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Pant Answer 3. Slip. In practically every home and workplace, there are electrical appliances and machinery. Although they are common and practical, they can also be quite dangerous. Thousands of people suffer amazed each year."
#reference_translations = ["When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Trip Answer 3. Slip. Electrical appliances and machinery are found in virtually every home and workplace. While they are common and convenient, they can also be quite dangerous. Thousands of people are shocked every year."]
#reference_text = "When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Trip Answer 3. Slip. Electrical appliances and machinery are found in virtually every home and workplace. While they are common and convenient, they can also be quite dangerous. Thousands of people are shocked every year."

candidate_translation = st.text_area("candidate_translation")
reference_translation = st.text_area("reference_translation")
st.button("Calculate Metrics", type="primary")

if st.button:
  reference_translations = [reference_translation]
  candidate_tokens = candidate_translation.split()
  references_tokens = [ref.split() for ref in reference_translations]
  st.write("Corpus BLEU Score:", round(corpus_bleu([references_tokens], [candidate_tokens]),2))
  st.write("Sentence BLEU Score:", round(sentence_bleu([reference_translation.split()], candidate_tokens),2))
  st.write("Meteor Score:", round(meteor([word_tokenize(candidate_translation)],word_tokenize(reference_translation)), 2))
  st.write("meteor_score Score:", round(meteor_score(candidate_translation, reference_translation),2))
