import os
import re
import textwrap
import requests
import time
import math
import subprocess
import json
from xml.etree.ElementTree import ParseError
from pydub import AudioSegment
from pydub.utils import mediainfo
import speech_recognition as sr
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import List, Dict, Any
from requests.exceptions import Timeout as TimeoutError  # Import TimeoutError

# Google Transcription API
# Milliseconds of audio to process per request to limit timeout errors
# Reduced for better recognition on potentially noisy audio
TRANSCRIPTION_CHUNK_SIZE = 15000  # 15 seconds

def get_transcript(video_url, language='en'):
    """Fetches the transcript from a YouTube video."""
    video_id = video_url.split("v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([language])
            transcript_data = transcript.fetch()
            return video_id, "\n".join([entry['text'] for entry in transcript_data])
        except NoTranscriptFound:
            print(f"No transcript in {language}. Attempting to find a generated one.")
            transcript = transcript_list.find_generated_transcript([language])
            transcript_data = transcript.fetch()
            return video_id, "\n".join([entry['text'] for entry in transcript_data])
        except ParseError:
            print("Parse error while fetching transcript. The transcript may be empty or invalid.")
            return video_id, None
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"Error: {e}")
        return video_id, None
    except Exception as e:
        print(f"Unexpected error fetching transcript: {e}")
        return video_id, None

def transcribe_audio(audio_path, temp_dir):
    print(f"Transcribes an audio file using Google's Web Speech API", flush=True)
    audio = AudioSegment.from_wav(audio_path)
    recognizer = sr.Recognizer()
    recognizer.operation_timeout = 120  # Set timeout in seconds for API requests
    transcript = ""
    print(f"Chunk length: {TRANSCRIPTION_CHUNK_SIZE}", flush=True)
    iterations = math.ceil(len(audio) / TRANSCRIPTION_CHUNK_SIZE)
    print(f"Number of chunks to process: {iterations}", flush=True)
    for i in range(0, len(audio), TRANSCRIPTION_CHUNK_SIZE):
        print(f"Processing chunk {math.ceil(i/TRANSCRIPTION_CHUNK_SIZE + 1)}... ")
        chunk = audio[i:i + TRANSCRIPTION_CHUNK_SIZE]
        # No normalization on chunks to avoid distortion
        temp_chunk_path = os.path.join(temp_dir, "temp_chunk.wav")
        chunk.export(temp_chunk_path, format="wav")
        with sr.AudioFile(temp_chunk_path) as source:
            audio_data = recognizer.record(source)
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                transcript += text + " "
                success = True
                break  # Success, exit retry loop
            except sr.UnknownValueError:
                print(f"Could not understand audio chunk starting at {i/1000} seconds", flush=True)
                break  # Do not retry on unintelligible audio
            except (sr.RequestError, TimeoutError) as e:
                print(f"Attempt {attempt+1}/{max_retries} failed for chunk at {i/1000}s: {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2)  # Short delay before retry
        if not success:
            print(f"Max retries exceeded or permanent error for chunk at {i/1000}s", flush=True)
        if os.path.exists(temp_chunk_path):
            os.remove(temp_chunk_path)
    return transcript.strip()

def cleanup(file_list):
    """Removes a list of files if they exist."""
    print("\nCleaning up temporary files...")
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)
            print(f" - Removed {f}")

def get_audio_stream_info(file_path):
    """Use ffprobe to get audio stream information."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", file_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
        if audio_stream:
            codec = audio_stream.get('codec_name', 'unknown')
            sample_rate = audio_stream.get('sample_rate', 'unknown')
            channels = audio_stream.get('channels', 1)
            bit_depth = audio_stream.get('bits_per_sample', 'unknown')
            return {
                'codec': codec,
                'sample_rate': int(sample_rate) if sample_rate != 'unknown' else 44100,
                'channels': channels,
                'bit_depth': bit_depth
            }
        else:
            print("No audio stream found in the file.")
            return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
        print(f"Error getting audio info: {e}")
        return None

def main():
    video_url = input("Enter the YouTube video URL: ")
    video_id, transcript = get_transcript(video_url)
    transcript_filename = f"{video_id}_transcript.txt"
    temp_files = []
    if transcript is None:
        print("Failed to fetch YouTube subtitles. Attempting audio transcription...")
        try:
            # Download audio
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': f'{video_id}.%(ext)s',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            audio_path = f"{video_id}.wav"
            temp_files.append(audio_path)
            optimized_audio_path = "optimized_audio.wav"
            temp_files.append(optimized_audio_path)

            # Get audio stream info
            audio_info = get_audio_stream_info(audio_path)
            if audio_info:
                print(f"Audio stream: codec={audio_info['codec']}, sample_rate={audio_info['sample_rate']}Hz, channels={audio_info['channels']}, bit_depth={audio_info['bit_depth']}")
            else:
                # Default
                audio_info = {'sample_rate': 44100, 'channels': 2, 'codec': 'unknown'}

            # Minimal FFmpeg extraction: No filters, keep stereo, original sample rate to avoid distortion
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-map", "0:a:0",  # Select first audio stream
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # Clean PCM
                "-ar", str(audio_info['sample_rate']),  # Keep original sample rate
                "-y",  # Overwrite output
                optimized_audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"FFmpeg extraction output: {result.stdout}")
            if result.stderr:
                print(f"FFmpeg warnings: {result.stderr}")

            # Load the WAV with pydub
            audio = AudioSegment.from_wav(optimized_audio_path)
            print(f"Loaded WAV: duration {len(audio)/1000:.2f}s, channels {audio.channels}, rate {audio.frame_rate}Hz")

            # Convert to mono by taking left channel only to avoid mixing artifacts
            if audio.channels > 1:
                print("Converting stereo to mono using left channel...")
                audio = audio.split_to_mono()[0]

            # Set to 16kHz for better speech recognition, no normalization to preserve original
            audio = audio.set_frame_rate(16000)

            # Mild boost if too quiet
            if audio.dBFS < -25:
                audio = audio + 3  # Gentle 3dB boost

            # Re-export the processed audio for transcription
            audio.export(optimized_audio_path, format="wav")

            # Debug: Check info
            info = mediainfo(optimized_audio_path)
            print(f"WAV Info: {info}")
            print(f"Final audio loudness: {audio.dBFS:.2f} dB")

            # If duration is 0 or very short, extraction failed—raise error
            if len(audio) == 0:
                raise ValueError("Exported WAV is empty—check FFmpeg setup or MP4 audio track.")

            transcript = transcribe_audio(optimized_audio_path, os.getcwd())

            print("Transcript from audio file:")
            print(transcript)
            with open(transcript_filename, "w") as file:
                file.write(transcript)
            print(f"Transcript saved to {transcript_filename}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed: {e.stderr}")
            cleanup(temp_files)
            return
        except Exception as e:
            print(f"Failed to download or transcribe audio: {e}")
            cleanup(temp_files)
            return
    else:
        print("Transcript from YouTube subtitles:")
        print(transcript)
        with open(transcript_filename, "w") as file:
            file.write(transcript)
        print(f"Transcript saved to {transcript_filename}")

    # Cleanup all temporary files
    cleanup(temp_files)

if __name__ == "__main__":
    main()