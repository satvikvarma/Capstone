import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the CountVectorizer, TF-IDF transformer, and Logistic Regression model
with open("count_vector.pkl", "rb") as f:
    vocab = pickle.load(f)
    load_vec = CountVectorizer(vocabulary=vocab)

load_tfidf = pickle.load(open("tfidf.pkl", "rb"))
load_model = pickle.load(open("logreg_model.pkl", "rb"))

# Function to predict a topic for custom text
def topic_predictor(text, target_names):
    X_new_count = load_vec.transform([text])
    X_new_tfidf = load_tfidf.transform(X_new_count)
    prediction = load_model.predict(X_new_tfidf)
    return target_names[prediction[0]], load_model.predict_proba(X_new_tfidf)[0]

def is_valid_input(input_text):
    """ Check if the input text is considered valid. """
    # Check if the text contains alphabetic characters
    if any(char.isalpha() for char in input_text):
        return True
    return False

# Streamlit app
def main():
    st.title("Customer Complaints Topic Classifier")

    # Input text box for user to enter complaint
    user_input = st.text_area("Enter your customer complaint here:")

    # Predict button to trigger the prediction
    if st.button("Predict Topic"):
        if not is_valid_input(user_input):
            st.write("Please enter the text in the correct format.")
        elif user_input.isdigit():
            st.write("Please enter your complaint in text format.")
        elif user_input:
            # Define target names
            target_names = ["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]
            
            # Get predicted topic and probabilities
            predicted_topic, predicted_probabilities = topic_predictor(user_input, target_names)

            # Check if predicted probabilities are below 0.60
            if all(prob < 0.60 for prob in predicted_probabilities):
                predicted_topic = "Not Valid. Please enter relevant text."
            st.write(f"Predicted Topic: {predicted_topic}")
        else:
            st.write("Please enter a complaint before predicting.")

if __name__ == "__main__":
    main()
