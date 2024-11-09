import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

# Load the TTS model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Function to generate a download link for the audio file
def get_audio_download_link(filename):
    href = f'<a href="{filename}">Download audio</a>'
    return href

# Streamlit app
st.title("Text-to-Speech")

# User input
text = st.text_area("Enter text to convert to speech")

if st.button("Generate Speech"):
    if text:
        # Generate speech
        inputs = processor(text=text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
       
        # Save the speech
        sf.write("speech.wav", speech.numpy(), samplerate=16000)
       
        # Display audio player
        st.audio("speech.wav", format="audio/wav")
        # Download link
        st.markdown('<a href="speech.wav">Download audio</a>', unsafe_allow_html=True)