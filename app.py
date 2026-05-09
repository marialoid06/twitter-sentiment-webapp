import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("Twitter Sentiment Analysis")
st.markdown("Enter a tweet to see the AI's prediction and the words that influenced it.")

user_input = st.text_area("Tweet Content", placeholder="Type your tweet here...", height=100)

if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Prediction & Confidence
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        probs = model.predict_proba(input_tfidf)[0]
        
        # Mapping results
        label_map = {1.0: "Positive", 0.0: "Neutral", -1.0: "Negative"}
        result = label_map[prediction]
        confidence = max(probs)

        pivot_words = ["however", "but", "although"]
        if any(word in user_input.lower().split() for word in pivot_words):
            st.warning("**Context Alert:** The AI detected a pivot word ('however', 'but'). The true sentiment might be more mixed than the math suggests!")
        
        if confidence < 0.20:
            st.warning(f"**Result: Inconclusive** (Confidence too low: {confidence:.1%})")
            st.markdown("*The AI is unsure about this tweet. It might contain mixed signals or lack strong emotional keywords.*")
        elif prediction == 1.0:
            st.success(f"**Result: Positive** ({confidence:.1%} confidence)")
        elif prediction == 0.0:
            st.info(f"**Result: Neutral** ({confidence:.1%} confidence)")
        else:
            st.error(f"**Result: Negative** ({confidence:.1%} confidence)")
        
        # 2. Keyword Impact Analysis
        st.subheader("Keyword Impact")
        
        # Get feature names and log probabilities
        words = user_input.lower().split()
        feature_names = vectorizer.get_feature_names_out()
        
        # Finding weights for the predicted class
        class_index = list(model.classes_).index(prediction)
        weights = model.feature_log_prob_[class_index]
        
        # Check which words in the tweet are in the vocabulary
        impact_list = []
        for word in words:
            if word in feature_names:
                word_idx = list(feature_names).index(word)
                impact_list.append({"Word": word, "Influence Score": weights[word_idx]})
        
        if impact_list:
            impact_df = pd.DataFrame(impact_list).sort_values(by="Influence Score", ascending=False)
            st.write("The AI gave these words the most 'weight' for its decision:")
            st.dataframe(impact_df, hide_index=True)
        else:
            st.write("No specific keywords were strong enough to highlight.")
            
    else:
        st.warning("Please enter some text first.")
st.markdown("---")
st.markdown("© 2026 Shweta Maria Loid. Licensed under the MIT License.")