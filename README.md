<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1>Voice-to-Text Summarizer via Transfer Learning and Semantic Representation</h1>

  <h2>Overview</h2>
  <p>This project involves building a <b>Voice-to-Text Summarizer</b> that converts spoken language into text and summarizes the resulting transcript using <b>transfer learning</b> and <b>semantic representation</b> techniques. The system employs models such as <b>Hubert</b> and <b>Wave2Vec</b> for automatic speech recognition (ASR) and <b>BART</b> and <b>BERT</b> for text summarization.</p>

  <h2>Objectives</h2>
  <ul>
    <li><b>Voice to Text Conversion</b>: Used <b>Hubert</b> and <b>Wave2Vec</b> models for transcribing speech into text from various audio formats (e.g., MP3, WAV).</li>
    <li><b>Text Summarization</b>: Applied <b>BERT</b> for extractive summarization and <b>BART</b> for abstractive summarization to condense the transcribed text.</li>
    <li><b>Real-Time Processing</b>: Integrated <b>Streamlit</b> to provide an interactive web interface for real-time voice input and summarization.</li>
  </ul>

  <h2>Data Preprocessing</h2>
  <p>Before feeding the audio files into the ASR models, several preprocessing steps were applied to improve the model’s generalization ability and performance:</p>
  <ul>
    <li><b>Noise Addition</b>: Slight noise was added to the audio to simulate real-world scenarios and enhance model robustness.</li>
    <li><b>Silence Removal</b>: Silence gaps were removed to ensure smoother transitions in the audio and improve transcription accuracy.</li>
    <li><b>Loudness Adjustment</b>: The loudness of the audio was increased to ensure clarity and consistency across different samples.</li>
  </ul>

  <h2>ASR Model Performance</h2>
  <p>We fine-tuned <b>Hubert</b> and <b>Wave2Vec</b> models on our dataset for automatic speech recognition (ASR). Below are the performance metrics:</p>
  <ul>
    <li><b>Hubert Performance</b>:
      <ul>
        <li>Average WER (normalized): 0.10</li>
        <li>Average Precision: 0.92</li>
        <li>Average Recall: 0.91</li>
        <li>Average F1 Score: 0.92</li>
        <li>Overall Word Accuracy: 90.26%</li>
      </ul>
    </li>
    <li><b>Wave2Vec Performance</b>:
      <ul>
        <li>Average WER (normalized): 0.13</li>
        <li>Average Precision: 0.88</li>
        <li>Average Recall: 0.88</li>
        <li>Average F1 Score: 0.88</li>
        <li>Overall Word Accuracy: 86.55%</li>
      </ul>
    </li>
  </ul>
  <p>Based on these results, <b>Hubert</b> outperformed <b>Wave2Vec</b>, providing higher accuracy and better overall performance for ASR tasks.</p>

  <h2>Text Summarization</h2>
  <p>After transcribing the audio to text, we fine-tuned the <b>BART</b> model for abstractive summarization and used <b>BERT</b> for extractive summarization:</p>
  <ul>
    <li><b>BART Performance (Abstractive Summarization)</b>:
      <ul>
        <li>eval_loss: 0.28</li>
        <li>eval_rouge1: 0.45</li>
        <li>eval_rouge2: 0.25</li>
        <li>eval_rougeL: 0.35</li>
        <li>eval_rougeLsum: 0.30</li>
      </ul>
    </li>
    <li><b>BERT Performance (Extractive Summarization)</b>:
      <ul>
        <li>Precision: 0.87</li>
        <li>Recall: 0.85</li>
        <li>F1 Score: 0.86</li>
      </ul>
    </li>
  </ul>

  <h2>Streamlit Interface</h2>
  <p>The project integrates a <b>Streamlit interface</b> that provides the following features for a user-friendly experience:</p>
  <ul>
    <li><b>Upload Audio</b>: Users can upload their voice recordings in <b>MP3</b>, <b>WAV</b>, or <b>FLAC</b> formats for transcription.</li>
    <li><b>Live Listening</b>: The interface allows users to record audio directly and listen to it in real-time before transcription.</li>
    <li><b>Two Types of Summarization</b>: After transcription, users are provided with two options for summarization:
      <ul>
        <li><b>Extractive Summarization</b> using <b>BERT</b>: Extracts key sentences from the transcription.</li>
        <li><b>Abstractive Summarization</b> using <b>BART</b>: Generates a condensed, reworded summary of the transcription.</li>
      </ul>
    </li>
    <li><b>Real-Time Summary</b>: Once the transcription is complete, the chosen summary type is generated and displayed instantly.</li>
  </ul>

  <h2>Technologies Used</h2>
  <ul>
    <li><b>Hubert</b>: Finetuned model used for speech-to-text conversion.</li>
    <li><b>Wave2Vec</b>: Alternative ASR model used for speech-to-text conversion.</li>
    <li><b>BERT</b>: Finetuned model used for extractive text summarization.</li>
    <li><b>BART</b>: Model used for abstractive text summarization.</li>
    <li><b>Python</b>: Programming language used for model development and optimization.</li>
    <li><b>Streamlit</b>: For building the interactive interface for uploading audio and viewing summaries.</li>
    <li><b>Libraries</b>: TensorFlow, Keras, Hugging Face Transformers, pandas, and Streamlit.</li>
  </ul>

 

</body>
</html>
