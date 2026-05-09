import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def clean_text_with_lemmas(text):
    # ... (previous cleaning steps like lowering and regex)
    tokens = text.split()
    # Convert each word to its root form
    lemmed_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(lemmed_tokens)

def clean_text(text):
    # 1. Lowercase and remove special characters/URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Stop word removal
    stop_words = set(stopwords.items('english'))
    filtered_text = [w for w in tokens if w not in stop_words]
    
    return " ".join(filtered_text)