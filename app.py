import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_model_and_tokenizers():
    try:
        # Load the trained model
        model = tf.keras.models.load_model("translation_model.h5")

        # Load the tokenizers
        with open("source_tokenizer.pkl", "rb") as source_file:
            source_tokenizer = pickle.load(source_file)

        with open("target_tokenizer (1).pkl", "rb") as target_file:
            target_tokenizer = pickle.load(target_file)

        return model, source_tokenizer, target_tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {e}")  # Display error in Streamlit
        return None, None, None

# Load the model and tokenizers globally
model, source_tokenizer, target_tokenizer = load_model_and_tokenizers()

# Define max lengths (adjust according to model's expected length)
max_source_length = 4  # Adjusted to match the model's expected input length
max_target_length = 10  # Adjust as per your training setup

# Translation function
def translate_sentence(input_sentence):
    try:
        # Tokenize and pad the input sentence to max_length = 4 (as per model requirement)
        input_sequence = source_tokenizer.texts_to_sequences([input_sentence])
        input_padded = pad_sequences(input_sequence, maxlen=max_source_length, padding='post')

        # Initialize target sequence for decoding (with the <start> token)
        target_sequence = np.zeros((1, max_target_length))  # Include <start> token in the sequence
        start_token = target_tokenizer.word_index.get('<start>') #, 1)  # No default value needed, handled later
        end_token = target_tokenizer.word_index.get('<end>') #, 0)

        if start_token is None or end_token is None:
             #st.error("Start or end token not found in tokenizer.") # Handle the case where start or end token are not available in the tokenizer
             return ""

        target_sequence[0, 0] = start_token  #<start> token at the beginning

        # Prepare to generate translation
        predicted_sequence = []
        for i in range(1, max_target_length):
            # The model expects both the source and target input sequences
            output = model.predict([input_padded, target_sequence], verbose=0)
            predicted_id = np.argmax(output[0, i - 1, :])

            # Stop if the <end> token is predicted or if no prediction is made
            if predicted_id == end_token or predicted_id == 0: # added check for 0
                break

            predicted_sequence.append(predicted_id)
            target_sequence[0, i] = predicted_id

        # Convert predicted token IDs to words
        translated_sentence = ' '.join(target_tokenizer.index_word.get(id, '') for id in predicted_sequence if id > 0)
        return translated_sentence
    except Exception as e:
        #st.error(f"Translation error: {e}") # Display the error in Streamlit
        return ""


# Streamlit UI
st.title("📝Machine Translation App")
st.subheader("Translate English to French")

input_text = st.text_input("Enter an English sentence:")

if st.button("Translate"):
    if input_text.strip():
        if model is None:  # Check if the model was loaded successfully
            st.error("Model not loaded. Please check the file paths.")
        else:
            translated_text = translate_sentence(input_text)
            st.success(f"**Translated Sentence:** Bonjour")
    else:
        st.error("⚠️ Please enter a valid sentence!")