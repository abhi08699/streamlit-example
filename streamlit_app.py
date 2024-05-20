import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import evaluate
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

candidate_translation = st.text_area("candidate_translation")
reference_translation = st.text_area("reference_translation")
st.button("Calculate Metrics", type="primary")

if st.button:
  reference_translations = [reference_translation]
  candidate_tokens = candidate_translation.split()
  references_tokens = [ref.split() for ref in reference_translations]
  st.write("Corpus BLEU Score:", round(corpus_bleu([references_tokens], [candidate_tokens]),2))
  st.write("Meteor Score:", round(meteor([word_tokenize(reference_translation)],word_tokenize(candidate_translation)), 2))
 # st.write("meteor_score Score:", round(meteor_score([references_tokens], [candidate_tokens]),2))
  bleu = evaluate.load('bleu')
  st.write("Evaluate blue score", round(bleu.compute(predictions=[candidate_translation], references=[reference_translation]),2))
