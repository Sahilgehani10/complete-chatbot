from transformers import pipeline
import librosa
import io
from utils import load_config, timeit
import os
import subprocess
import shutil

config = load_config()

def convert_webm_to_wav_ffmpeg(audio_bytes):
    try:
        # Check if FFmpeg is available
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found in system PATH")

        # Create a temp directory to store files
        os.makedirs('temp', exist_ok=True)
        
        # Full paths for temp files
        temp_webm = os.path.join('temp', 'temp_audio.webm')
        temp_wav = os.path.join('temp', 'temp_audio.wav')

        # Save the WebM bytes to a file
        with open(temp_webm, "wb") as f:
            f.write(audio_bytes)

        # Use FFmpeg to convert WebM to WAV with full path
        result = subprocess.run(
            [ffmpeg_path, "-fflags", "+igndts", "-i", temp_webm, "-c:a", "pcm_s16le", temp_wav],
            capture_output=True,
            text=True
        )

        # Check for conversion errors
        if result.returncode != 0:
            print("FFmpeg Error Output:", result.stderr)
            raise RuntimeError(f"FFmpeg failed to convert WebM to WAV: {result.stderr}")

        # Read the WAV file back into memory
        with open(temp_wav, "rb") as f:
            wav_data = f.read()

        wav_io = io.BytesIO(wav_data)

        # Clean up the temp files
        os.remove(temp_webm)
        os.remove(temp_wav)

        return wav_io

    except Exception as e:
        print(f"Conversion error: {e}")
        raise

def convert_bytes_to_array(audio_bytes):
    try:
        audio_bytes_io = io.BytesIO(audio_bytes)
        audio, sample_rate = librosa.load(audio_bytes_io, sr=16000)
    except Exception as e:
        print(f"Initial audio load error: {e}")
        try:
            wav_io = convert_webm_to_wav_ffmpeg(audio_bytes)
            audio, sample_rate = librosa.load(wav_io, sr=16000)
        except Exception as conv_error:
            print(f"Conversion fallback error: {conv_error}")
            raise

    print(f"Sample rate: {sample_rate}")
    return audio

@timeit
def transcribe_audio(audio_bytes):
    device = "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )   

    try:
        audio_array = convert_bytes_to_array(audio_bytes)
        prediction = pipe(audio_array, batch_size=1)["text"]
        return prediction
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Could not transcribe audio"