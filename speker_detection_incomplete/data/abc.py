from pydub import AudioSegment
import os

def split_audio(file_path, output_dir, duration_ms=1000):
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)
        audio_length = len(audio)
        print(f"Audio duration: {audio_length / 1000:.2f} seconds")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Split and save audio segments
        for i in range(0, audio_length, duration_ms):
            segment = audio[i:i + duration_ms]
            segment_file_name = f"{output_dir}/{(i // duration_ms) + 1}.wav"  # Updated file naming
            segment.export(segment_file_name, format="wav")
            print(f"Saved {segment_file_name}")

        print("Audio splitting completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specify the input WAV file and output directory
    input_file = "test.wav"  # Replace with your WAV file
    output_directory = "output_segments"

    split_audio(input_file, output_directory)

