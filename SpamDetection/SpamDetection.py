import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load dataset
data = pd.read_csv(r"C:\Users\kumar\Desktop\spam.csv")

# Clean dataset
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split data
X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer
cv = CountVectorizer(stop_words='english')
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

# Streamlit UI
st.header('📩 Spam Detector')

input_mess = st.text_input('Enter your message:')
if st.button('Validate'):
    output = predict(input_mess)
    if output == 'Spam':
        st.markdown("🚨 **SPAM**")
    else:
        st.markdown("✅ **NOT SPAM**")
