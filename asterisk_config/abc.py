import os
import subprocess
import logging
from gtts import gTTS

'''


	this file is to change the out.gsm file, out.gsm is played when the call is started, you can modify this as you wish. 



'''



# Set up logging
logging.basicConfig(
    filename='/pathToProject/astrisk_config/process_audio.log', 
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def text_to_speech(text, output_path):
    logging.info(f"Starting text-to-speech conversion for text: {text}")

    try:
        tts = gTTS(text=text, lang='en')
        temp_mp3_path = "/pathToProject/astrisk_config/temp_audio.mp3"
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
    output_gsm_path = "/pathToProject/astrisk_config/out.gsm"
    
    logging.info("Process started.")
    
    # Step 1: Convert predefined text to speech in GSM format
    response_text = "Hello, user, what is your query?"
    text_to_speech(response_text, output_gsm_path)
    
    logging.info("Text-to-speech conversion completed successfully.")
    logging.info("Process completed.")

