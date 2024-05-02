import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the CountVectorizer, TF-IDF transformer, and Logistic Regression model
with open("count_vector.pkl", "rb") as f:
    vocab = pickle.load(f)
    load_vec = CountVectorizer(vocabulary=vocab)

load_tfidf = pickle.load(open("tfidf.pkl","rb"))
load_model = pickle.load(open("logreg_model.pkl","rb"))

# Function to predict a topic for custom text
def topic_predictor(text):
    target_names = ["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]
    X_new_count = load_vec.transform([text])
    X_new_tfidf = load_tfidf.transform(X_new_count)
    prediction = load_model.predict(X_new_tfidf)
    return target_names[prediction[0]]

# Streamlit app
def main():
    st.title("Customer Complaints Topic Classifier")

    # Input text box for user to enter complaint
    user_input = st.text_area("Enter your customer complaint here:")

    # Predict button to trigger the prediction
    if st.button("Predict Topic"):
        if user_input:
            # Get predicted topic
            predicted_topic = topic_predictor(user_input)

            # Display the predicted topic
            st.write(f"Predicted Topic: {predicted_topic}")

if __name__ == "__main__":
    main()
