# Twitter Sentiment Analysis Web Application 🐦

An interactive machine learning application built with Streamlit that analyzes the sentiment of tweets (Positive, Neutral, or Negative).

## Features
* **Custom ML Model:** Powered by a Multinomial Naive Bayes classifier trained on over 160,000 tweets.
* **Smart Preprocessing:** Utilizes NLTK Lemmatization and Bigrams to understand context and word pairs.
* **Transparent AI:** Displays a "Keyword Impact" table to show exactly which words influenced the model's decision.
* **Confidence Guardrails:** Automatically flags weak predictions (< 50% confidence) as "Inconclusive" to maintain reliability.

## Tech Stack
* Python
* Scikit-Learn (TF-IDF, Naive Bayes)
* NLTK (WordNetLemmatizer)
* Pandas
* Streamlit

## Usage
Simply type a phrase into the text box and the AI will evaluate the emotional tone, providing a confidence score and a breakdown of the mathematical weights assigned to your words.
