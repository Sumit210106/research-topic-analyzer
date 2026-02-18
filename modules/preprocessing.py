import spacy 
import string 
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Text Preprocessing Pipeline:
    1. Lowercasing
    2. Tokenization (spaCy)
    3. Stopword removal (spaCy built-in)
    4. Punctuation removal
    5. Lemmatization (POS-aware)
    6. Remove non-alphabetic tokens
    """    
    text = text.lower()
    
    doc = nlp(text)
    
    cleaned_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]
    
    return ' '.join(cleaned_tokens)

    