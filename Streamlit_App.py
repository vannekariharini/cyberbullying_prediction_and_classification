import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk_packages = {
    'stopwords': 'corpora/stopwords',
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet'
}

for package, path in nltk_packages.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package)


# Load the saved XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_classifier = pickle.load(model_file)

# Load the saved CountVectorizer
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)



# Preprocessing setup
lemma = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update([ 'the', 'a', 'an','of','im', 'wa', 'p', 't', 's', 'o', 'e', 'like','and'])

# Function to clean text (same as used in training)
def clean_text(text):
    pattern = re.compile(r"(#[A-Za-z0-9]+|@[A-Za-z0-9]+|https?://\S+|www\.\S+|\S+\.[a-z]+|RT @)")
    text = pattern.sub('', text)
    text = " ".join(text.split())
    text = text.lower()
    text = " ".join([lemma.lemmatize(word) for word in word_tokenize(text)])
    remove_punc = re.compile(r"[%s]" % re.escape(string.punctuation))
    text = remove_punc.sub('', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

# Define function to predict tweet safety
def predict_tweet(tweet):
    # Clean and preprocess the new input text
    cleaned_new_text = [clean_text(tweet)]
    
    # Transform the new input text using the fitted CountVectorizer
    new_text_vectorized = count_vectorizer.transform(cleaned_new_text)
    
    # Predict the label for the new input text
    predicted_label = xgb_classifier.predict(new_text_vectorized)
    
    # Map the predicted label back to the corresponding class name
    label_map = {0: 'not_cyberbullying', 1: 'gender', 2: 'religion', 3: 'other_cyberbullying', 4: 'age', 5: 'ethnicity'}
    predicted_class = label_map[predicted_label[0]]
    
    return predicted_class

def main():
    # Set page heading
    st.markdown("<h1 style='text-align: center; font-size:36px; color: #22636f; font-weight:normal;'>PREDICTION AND CLASSIFICATION OF CYBERBULLYING TWEETS</h1><br/>", unsafe_allow_html=True)
    # Text area for user to type tweet
    
    tweet_input = st.text_area(r"$\textsf{\Large Enter the tweet}$", height=150)
    
    # Button to trigger prediction
    if st.button(r"$\textsf{\large Predict}$"):
        if tweet_input.strip() == "":
            st.warning("Please enter the tweet.")
        else:
            # Call function to predict tweet safety
            prediction = predict_tweet(tweet_input)
            st.write(f"The tweet is classified as: {prediction}")

if __name__ == "__main__":
    main()
