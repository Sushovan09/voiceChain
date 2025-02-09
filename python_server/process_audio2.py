from transformers import pipeline
import tensorflow.keras
import speech_recognition as sr
from gtts import gTTS
import os
import subprocess
import json
import logging
import requests
import speaker  # Assuming speaker is a module that has the function to predict speaker
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




logging.basicConfig(
    filename='/---projectPath---/python_server/test_log.log',  # Change to your desired log file name
    level=logging.DEBUG,  # Capture debug messages and higher
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.debug("This is a debug message.")
logging.info("This is an info message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")



# Local LLM API configuration
local_llm_url = "http://localhost:11434/api/generate"

# Load BERT emotion detection pipeline
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="tf")

def detect_emotion(text):
    """Detect emotion using BERT pipeline."""
    logging.info(f"Detecting emotion for text: {text}")
    try:
        emotions = emotion_pipeline(text)
        dominant_emotion = max(emotions, key=lambda x: x['score'])['label']
        logging.info(f"Detected emotion: {dominant_emotion}")
        return dominant_emotion
    except Exception as e:
        logging.error(f"Error detecting emotion: {e}")
        return "neutral"

def query_llm(text, emotion):
    """Query the local LLM API and return the response."""
    logging.info(f"Querying LLM with: {text}")

    enhanced_prompt = (
        f"This is a user query with the detected emotion '{emotion}'. "
        "The user is interacting with an intelligent IVR system. "
        f"Query: {text}\n"
        "Provide a clear and accurate response."
    )

    payload = {
        "model": "llama3",  # Specify your model name
        "prompt": enhanced_prompt
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send the POST request with streaming enabled
        response = requests.post(local_llm_url, json=payload, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Initialize an empty string to collect the response
        response_text = ""

        # Iterate over the streamed response chunks
        for chunk in response.iter_lines():
            if chunk:  # Skip empty keep-alive chunks
                try:
                    # Assume each chunk is JSON and decode it
                    data = json.loads(chunk.decode("utf-8"))
                    # Extract the text portion of the chunk
                    if "response" in data:
                        response_text += data["response"]
                except json.JSONDecodeError:
                    # Handle cases where the chunk is not valid JSON
                    logging.warning(f"Non-JSON chunk received: {chunk.decode('utf-8')}")
                    response_text += chunk.decode("utf-8")

        # Remove any unintended newlines or formatting issues
        response_text = response_text.strip()
        logging.info(f"Received response from LLM: {response_text}")
        return response_text
                

    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying LLM: {e}")
        return None

def transcribe_audio(file_path):
    logging.info(f"Starting transcription for file: {file_path}")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            logging.info("Audio file loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        return None

    try:
        text = recognizer.recognize_google(audio_data)
        logging.info(f"Transcription successful. Transcribed text: {text}")

        # Save the transcription as query in the file
        text_file_path = file_path.replace(".wav", ".txt")
        with open(text_file_path, "w") as text_file:
            text_file.write(f"<query> {text}\n")
            logging.info(f"Transcribed text saved to {text_file_path}.")

        return text, text_file_path

    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return None, None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None

'''
def predict_speaker(file_path):
    """Use speaker detection to predict the speaker from the audio file."""
    try:
        speaker_name = speaker.predict_speaker_from_audio(file_path)
        logging.info(f"Detected speaker: {speaker_name}")
        return speaker_name
    except Exception as e:
        logging.error(f"Error detecting speaker: {e}")
        return "unknown"
'''

def text_to_speech(text, output_path):
    logging.info(f"Starting text-to-speech conversion for text: {text}")

    try:
        tts = gTTS(text=text, lang='en')
        temp_mp3_path = "/---projectPath---/python_server/temp_audio.mp3"
        tts.save(temp_mp3_path)
        logging.info("Text converted to speech and saved as temp_audio.mp3.")

        if not os.path.exists(temp_mp3_path):
            logging.error("Temporary MP3 file does not exist after saving.")
            return

        # Convert mp3 to GSM using ffmpeg
        logging.info(f"Running ffmpeg to convert {temp_mp3_path} to {output_path}")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_mp3_path, "-ar", "8000", "-ac", "1", output_path],
            capture_output=True, text=True
        )
        logging.info(f"ffmpeg output: {result.stdout}")
        logging.info(f"ffmpeg errors: {result.stderr}")

        if result.returncode != 0:
            logging.error(f"ffmpeg failed with return code {result.returncode}")
            return

        logging.info(f"Converted temp_audio.mp3 to GSM format and saved as {output_path}.")
        os.remove(temp_mp3_path)  # Clean up
        logging.info("Temporary MP3 file removed.")

    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logging.error("Usage: python3 process_audio.py <input_wav_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_gsm_path = "/---projectPath---/python_server/output_audio.gsm"

    logging.info("Process started.")

    # Step 1: Transcribe audio to text
    query_text, text_file_path = transcribe_audio(file_path)

    if query_text:
        # Step 2: Detect the speaker from the audio
        #speaker_name = predict_speaker(file_path)
        # Step 3: Detect emotion in the transcribed text
        emotion = detect_emotion(query_text)

        # Step 4: Query the LLM with the transcribed text, speaker name, and emotion
        response_text = query_llm(query_text, emotion)

        if response_text:
            # Step 5: Save the query and the LLM response in the same text file
            with open(text_file_path, "a") as text_file:
                text_file.write(f"<response> {response_text}\n")  # Save the LLM response
                logging.info(f"Query and response saved to {text_file_path}.")

            # Step 6: Convert the LLM response to speech
            text_to_speech(response_text, output_gsm_path)
            logging.info("Text-to-speech conversion for the response completed successfully.")
        else:
            logging.warning("No response from LLM.")
    else:
        logging.warning("No valid text to query LLM.")

    logging.info("Process completed.")

