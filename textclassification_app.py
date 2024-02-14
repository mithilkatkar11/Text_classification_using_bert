import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

# Load pre-trained BERT model and tokenizer with 3 labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to('cpu')
model.eval()

# Function to classify text
def classify_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    # Get predicted class (index with highest probability)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
    return predicted_class, probabilities

# Streamlit app
st.title('Text Classification with BERT')
st.write('Enter some text for classification')

user_input = st.text_area("Input text here:", "")

if st.button('Classify'):
    if user_input.strip() == "":
        st.error("Please enter some text.")
    else:
        predicted_class, probabilities = classify_text(user_input)
        st.write(f'Predicted Class: {predicted_class}')
        st.write(f'Probabilities: {probabilities}')

        # Add a basic explanation for the user
        if predicted_class == 0:
            st.write('Class 0 might represent a negative sentiment.')
        elif predicted_class == 1:
            st.write('Class 1 might represent a neutral sentiment.')
        elif predicted_class == 2:
            st.write('Class 2 might represent a strong positive sentiment.')