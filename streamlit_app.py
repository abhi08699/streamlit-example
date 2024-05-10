import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import nltk as nltk
from nltk.translate.bleu_score import corpus_bleu
# Example usage:
candidate_translation = "When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Pant Answer 3. Slip. In practically every home and workplace, there are electrical appliances and machinery. Although they are common and practical, they can also be quite dangerous. Thousands of people suffer amazed each year."
reference_translations = ["When there is a lack of adherence between the feet and the surface of a step, it is: Answer 1. Fall Answer 2. Trip Answer 3. Slip. Electrical appliances and machinery are found in virtually every home and workplace. While they are common and convenient, they can also be quite dangerous. Thousands of people are shocked every year."]

# Calculate BLEU score
#bleu_score = calculate_bleu(candidate_translation, reference_translations)
candidate_tokens = candidate_translation.split()
references_tokens = [ref.split() for ref in reference_translations]
bleu_score =  corpus_bleu([references_tokens], [candidate_tokens])
print("BLEU Score:", round(bleu_score,2))
