import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import wavio
import time
import torch
import torchaudio 
from transformers import HubertForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer, pipeline
from summarizer import Summarizer  
from io import BytesIO

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Function to save the uploaded audio file
def save_uploaded_audio(uploaded_file):
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

# Function to record live audio
def record_live_audio(duration=30, samplerate=44100, channels=1):  # Set to mono (1 channel)
    st.write("Recording live audio...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    st.write("Recording finished. Processing...")

    # Save recorded audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavio.write(temp_file.name, samplerate, audio_data.astype(np.int16))  # Ensure the audio is in int16 format
    
    return temp_file.name

# Function to transcribe audio using Hubert model (Optimized)
def transcribe_audio(audio_path):
    # Load Hubert model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)  # Correct model class for Hubert

    # Load and preprocess the audio file (Using torchaudio for better performance)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert stereo to mono if needed

    # Resample audio to the model's expected sample rate (16kHz for Hubert)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Process the waveform and prepare inputs for the model
    inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", padding=True).to(device)  # Move input tensors to GPU
    
    # Perform inference and decode the result
    logits = model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

# Function to perform extractive summarization using Hugging Face's BERT-based model
def extractive_summary(text):
    # Initialize the BERT extractive summarizer
    bert_model = Summarizer()

    # Perform extractive summarization
    summary = bert_model(text)  # Perform extractive summarization

    return summary 

# Function to summarize text using BART model (abstractive summarization)
def abstractive_summary(text):
    # Load BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)  # Move model to GPU if available
    
    # Tokenize the input text
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True).to(device)  # Move input tensors to GPU
    summary_ids = model.generate(inputs['input_ids'], max_length=200, num_beams=4, early_stopping=True)
    
    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Streamlit UI
st.title("Voice to Text Summarizer")

# Choose input method
option = st.radio("Choose audio input method:", ("Upload Audio File", "Live Listen"))

# Summarization Type Selection (Extractive or Abstractive)
summary_type = st.selectbox("Choose summarization type:", ("Abstractive", "Extractive"))

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "flac"])
    if uploaded_file is not None:
        save_uploaded_audio(uploaded_file)
        st.write("Audio file uploaded successfully!")
        
        # Pass audio to Hubert model for speech-to-text conversion
        st.write("Processing the audio...")
        transcription = transcribe_audio("uploaded_audio.wav")
        
        # Perform summarization based on user selection
        if summary_type == "Extractive":
            summary = extractive_summary(transcription)
        else:
            summary = abstractive_summary(transcription)
        
        st.subheader("Transcription")
        st.write(transcription)
        
        st.subheader(f"Summary ({summary_type})")
        st.write(summary)

elif option == "Live Listen":
    st.write("Click to start live listening...")
    if st.button("Start Listening"):
        # Record live audio for 30 seconds
        audio_file = record_live_audio(duration=30)
        st.write(f"Live audio recorded and saved as {audio_file}")
        
        # Pass audio to Hubert model for speech-to-text conversion
        st.write("Processing the audio...")
        transcription = transcribe_audio(audio_file)
        
        # Perform summarization based on user selection
        if summary_type == "Extractive":
            summary = extractive_summary(transcription)
        else:
            summary = abstractive_summary(transcription)
        
        st.subheader("Transcription")
        st.write(transcription)
        
        st.subheader(f"Summary ({summary_type})")
        st.write(summary)

# Display processing status
if st.button("Process Audio"):
    st.write("Processing the audio... Please wait.")
    time.sleep(3)  # Simulate processing time
    st.write("Audio processed! Summary: ...")
