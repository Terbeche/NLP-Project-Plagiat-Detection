import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# Header section
st.title("Plagiarism Detection App")
st.markdown("""
**أعضاء الفريق**:
- Latti Othmane
- Bedjeboudja Anas
- Mahi Tani Issam
- Terbeche Mostefa
- Tahraoui Nour El Houda 
- Ziani Fatiha

**تحت إشراف الدكتور الأستاذ**:
- Dr. ABDERRAHIM MOHAMMED ALAEDDINE
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])  # Remove special characters
    text = ' '.join(text.split())
    return text

st.title("Plagiarism Detection NLP Project")

st.sidebar.header("Settings")
max_sequence_length = st.sidebar.number_input("Max Sequence Length", value=100, min_value=10, step=10)

st.header("Upload or Enter Text")
uploaded_file = st.file_uploader("Upload a Text File", type=["txt"])
input_text = st.text_area("Or Paste Text Below:")

model = load_model()
tokenizer = load_tokenizer()

if st.button("Detect Plagiarism"):
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
    elif not input_text.strip():
        st.warning("Please upload a file or enter text!")
        st.stop()
    
    preprocessed_text = preprocess_text(input_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence)
    st.subheader("Prediction Result")
    if prediction[0][0] > 0.5:
        st.error("Plagiarism Detected!")
    else:
        st.success("No Plagiarism Detected.")
    
    st.write(f"Confidence Score: {prediction[0][0]:.4f}")
