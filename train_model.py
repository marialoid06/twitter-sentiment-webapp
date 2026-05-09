import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import re

# Initialize resources
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def advanced_clean(text):
    text = str(text).lower()
    # Remove everything except letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    # Lemmatize each word
    lemmed_words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmed_words)

# 1. Load Data First
print("Loading data...")
df = pd.read_csv('Twitter_Data.csv')
df.dropna(inplace=True)

# 2. Apply Cleaning
print("Cleaning and Lemmatizing...")
df['clean_text'] = df['clean_text'].apply(advanced_clean)

# 3. Split
X = df['clean_text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize with Bigrams
print("Vectorizing text with n-grams...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train
print("Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Output results
print("\nClassification Report:")
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {model.score(X_test_tfidf, y_test):.2%}")

# 6. Save
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and Vectorizer saved!")