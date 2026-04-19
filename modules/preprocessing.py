import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK datasets are downloaded
for dataset in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{dataset}' if 'punkt' in dataset else f'corpora/{dataset}')
    except LookupError:
        nltk.download(dataset, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Text Preprocessing Pipeline using pure NLTK:
    1. Lowercasing
    2. Tokenization 
    3. Stopword removal 
    4. Punctuation removal
    5. Lemmatization 
    6. Remove non-alphabetic tokens
    """    
    text = str(text).lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
        and token not in string.punctuation
        and token.isalpha()
    ]
    
    return ' '.join(cleaned_tokens)