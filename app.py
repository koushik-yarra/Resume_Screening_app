import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Download required NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the saved model, label encoder, and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Function to clean the input resume text
def clean(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'-', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return " ".join(words)



# Streamlit app

st.title('RESUME SCREENING APP')
st.write('This app predicts the category of the resume')
uploaded_file = st.file_uploader("Choose a file", type=['txt','pdf', 'docx']) # Upload a file
if uploaded_file is not None:
    try:
        # Assuming you have a way to extract text from the PDF
        # Replace this with your PDF text extraction method
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages: # Loop through all pages
            text += page.extract_text() # Extract text from each page
        if st.button('predict'):
            cleaned_text = clean(text) # Clean the text

        # Vectorize the input text
            input_vector = cv.transform([cleaned_text]) 

        # Make prediction
            prediction = model.predict(input_vector)

        # Inverse transform the prediction to get category label
            category = le.inverse_transform(prediction)[0]

            st.write("Predicted Category:", category)
    except Exception as e:
            st.write("Error processing file:", str(e))

