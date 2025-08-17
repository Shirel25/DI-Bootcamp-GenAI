# import sys
# print(sys.executable)

data = {
    'Review': [
        'At McDonald\'s the food was ok and the service was bad.',
        'I would not recommend this Japanese restaurant to anyone.',
        'I loved this restaurant when I traveled to Thailand last summer.',
        'The menu of Loving has a wide variety of options.',
        'The staff was friendly and helpful at Google\'s employees restaurant.',
        'The ambiance at Bella Italia is amazing, and the pasta dishes are delicious.',
        'I had a terrible experience at Pizza Hut. The pizza was burnt, and the service was slow.',
        'The sushi at Sushi Express is always fresh and flavorful.',
        'The steakhouse on Main Street has a cozy atmosphere and excellent steaks.',
        'The dessert selection at Sweet Treats is to die for!'
    ]
}


import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# ================================
# ========== Question 1 ==========
# ================================

def preprocess_text(text):
    # 1. Convert to lower case
    new_text = text.lower()

    # 2. Tokenization
    doc = nlp(new_text)

    cleaned_tokens = []

    for token in doc:
        # 3. Remove punctuation and stopwords
        if token.text in string.punctuation:
            continue
        if token.is_stop:
            continue

        # 4. Lemmatize the token (get base form)
        lemma = token.lemma_

        # Optional: remove empty tokens or spaces
        if lemma.strip() != "":
            cleaned_tokens.append(lemma)

    return cleaned_tokens


print("=== Question 1 ===")
# Test the function on each review
for i, review in enumerate(data['Review']):
    print(f"Review {i+1}: {preprocess_text(review)}")



# ================================
# ========== Question 2 ==========
# ================================

import pandas as pd

df = pd.DataFrame(data) 
df['Clean_Review'] = df['Review'].apply(preprocess_text)

print("\n=== Question 2 ===")
print(df.head())

# ================================
# ========== Question 3 ==========
# ================================

def perform_ner(text):
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    
    return entities

print("\n=== Question 3 ===")
for i, review in enumerate(df['Review']):
    print(f"Review {i+1}: {perform_ner(review)}")


# ================================
# ========== Question 4 ==========
# ================================

import nltk
from nltk import pos_tag, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
def perform_pos_tagging(text):
    # 1. Tokenisation (NLTK here)
    tokens = word_tokenize(text)
    
    # 2. POS tagging
    pos_tags = pos_tag(tokens)
    
    return pos_tags

print("\n=== Question 4 ===")
for i, review in enumerate(df['Review']):
    print(f"Review {i+1}: {perform_pos_tagging(review)}")


# ================================
# =========== Question 5 =========
# ================================

# 1) Helper: when applying NER/POS on cleaned tokens, we need a text again
def tokens_to_text(tokens):
    """
    Join a list of tokens into a single string so that NER/POS
    can be run on text (spaCy/NLTK expect text, not token lists).
    """
    return " ".join(tokens)

# 2) Apply NER on raw vs. cleaned text
df["NER_Raw"] = df["Review"].apply(perform_ner)
df["NER_Clean"] = df["Clean_Review"].apply(lambda toks: perform_ner(tokens_to_text(toks)))

# 3) Apply POS on raw vs. cleaned text
df["POS_Raw"] = df["Review"].apply(perform_pos_tagging)
df["POS_Clean"] = df["Clean_Review"].apply(lambda toks: perform_pos_tagging(tokens_to_text(toks)))

# 4) Pretty print a side-by-side comparison for analysis
print("\n=== Question 5: NER & POS comparison (Raw vs Clean) ===")
for i, row in df.iterrows():
    print(f"\n--- Review {i+1} ---")
    print("Raw text:   ", row["Review"])
    print("Clean text: ", tokens_to_text(row["Clean_Review"]))
    print("NER Raw:    ", row["NER_Raw"])
    print("NER Clean:  ", row["NER_Clean"])
    print("POS Raw:    ", row["POS_Raw"])
    print("POS Clean:  ", row["POS_Clean"])
