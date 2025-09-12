import os
import re
import textwrap
import requests
import time
import math
import argparse
import subprocess
import json
from xml.etree.ElementTree import ParseError
from pydub import AudioSegment
from pydub.utils import mediainfo
import speech_recognition as sr
from typing import List, Dict, Any

# Google Transcription API
# Milliseconds of audio to process per request to limit timeout errors
# Reduced for better recognition on potentially noisy audio
TRANSCRIPTION_CHUNK_SIZE = 15000  # 15 seconds

# A default word limit to trigger chunking for summarization.
# This is a safe estimate for the llama3 8B model's ~8k token context window.
DEFAULT_WORD_LIMIT = 5000
CHUNK_SIZE = 2000

def check_ollama() -> bool:
    """Checks if the Ollama server is running and the llama3 model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(model["name"].startswith("llama3") for model in models):
                print("Ollama server running with llama3 model available.")
                return True
            else:
                print("Ollama server running, but 'llama3' model not found.")
                print("Please run 'ollama pull llama3' to download it.")
                return False
        else:
            print(f"Ollama server returned status {response.status_code}.")
            return False
    except requests.ConnectionError:
        print("Error: Ollama server not running. Please start it with 'ollama serve'.")
        return False

def chunk_text(text: str, max_words: int = CHUNK_SIZE) -> List[str]:
    """Splits text into chunks of a maximum word count."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_with_ollama(text: str, is_chunk: bool = False) -> (str, List[str]):
    """
    Sends text to Ollama for summarization. Handles large text by chunking.

    If is_chunk is True, it returns only the summary paragraph for a single chunk.
    Otherwise, it returns the final executive summary and key points.
    """
    word_count = len(re.findall(r'\b\w+\b', text))

    # Handle large documents by chunking and recursively summarizing
    if word_count > DEFAULT_WORD_LIMIT and not is_chunk:
        print(f"Document is large ({word_count} words). Splitting into chunks...", flush=True)
        chunks = chunk_text(text, max_words=CHUNK_SIZE)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...", flush=True)
            chunk_summary, _ = summarize_with_ollama(chunk, is_chunk=True)
            chunk_summaries.append(chunk_summary)

        combined_text = "\n\n".join(chunk_summaries)
        print("Generating final summary from chunk summaries...", flush=True)
        return summarize_with_ollama(combined_text)

    # Base case: summarize a single chunk or a small document
    if is_chunk:
        prompt = textwrap.dedent(f"""\
            Summarize the following document chunk in detail. The summary should be a comprehensive paragraph covering all key aspects.

            ---
            DOCUMENT CHUNK:
            {text}
            ---

            Please respond with only the summary paragraph.
            """)
    else:
        # Final summary prompt
        prompt = textwrap.dedent(f"""\
            You are a helpful assistant. Your task is to extract a detailed executive summary and a comprehensive list of key points from the provided document text.

            ---
            DOCUMENT TEXT:
            {text}
            ---

            Please respond with a detailed paragraph (200-400 words) for the executive summary, followed by a list of key points. Make the executive summary thorough and insightful, covering main themes, arguments, and conclusions. For key points, provide up to 30 detailed bullet points, each elaborating on important elements with context or explanations where relevant. Use the following format:

            Executive Summary:
            [Write the detailed summary paragraph here.]

            Key Points:
            - [List the first key point here, with detail.]
            - [List the second key point here, with detail.]
            - [Continue with up to 30 key points, using bullet points.]
            """)

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes

        data = response.json()
        raw_output = data["response"].strip()
        print(f"\n--- Raw Ollama Output for Parsing ---\n{raw_output}\n-----------------------------------\n", flush=True)

        # If summarizing a chunk, just return the raw output.
        if is_chunk:
            return raw_output, []

        # New, more robust parsing logic
        summary_start_marker = "Executive Summary:"
        key_points_start_marker = "Key Points:"

        summary_paragraph = ""
        key_points = []

        # Use regex to find and split the sections
        match = re.search(f"(?:^|\n\s*){re.escape(summary_start_marker)}(.*?)(?:^|\n\s*){re.escape(key_points_start_marker)}(.*)", raw_output, re.DOTALL | re.IGNORECASE)

        if match:
            # Extract and clean up the summary paragraph
            summary_paragraph = match.group(1).strip()
            summary_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', summary_paragraph, flags=re.MULTILINE).strip()

            # Extract and parse the key points
            points_text = match.group(2).strip()
            points_list = re.findall(r'^\s*(?:-|\*|•|\d+\.)\s*(.*?)$', points_text, re.MULTILINE)
            key_points = [re.sub(r'^\s*\*+\s*(.*?)\s*\*+\s*$', r'\1', point).strip() for point in points_list if point.strip()]
        else:
            # Fallback for when key points are not found, but a summary is
            summary_match = re.search(f"(?:^|\n\s*){re.escape(summary_start_marker)}(.*)", raw_output, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary_paragraph = summary_match.group(1).strip()
                summary_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', summary_paragraph, flags=re.MULTILINE).strip()


        return summary_paragraph, key_points

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {e}", flush=True)
        return "", []
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}", flush=True)
        return "", []

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
    """Main function to orchestrate the transcription and summarization process for an audio file."""
    parser = argparse.ArgumentParser(
        description="Transcribes and summarizes an audio file (MP3, WAV, MP4, or MOV) using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        Example usage:
          python summarizeAudio.py --file path/to/audio.mp3
          python summarizeAudio.py --file path/to/audio.wav -k
          python summarizeAudio.py --file path/to/video.mp4
          python summarizeAudio.py # Will prompt for a file path
        ''')
    )
    parser.add_argument("--file", type=str,
                        help="The path to the MP3, WAV, MP4, or MOV audio/video file to process.")
    parser.add_argument("-k", "--keep-transcript", action="store_true", default=False,
                        help="Retain the raw transcript in a text file.")
    args = parser.parse_args()

    audio_file = args.file
    # If the file path is not provided as a command-line argument, prompt the user for it.
    if audio_file is None:
        audio_file = input("Please enter the path to the MP3, WAV, MP4, or MOV audio/video file: ")

    keep_transcript = args.keep_transcript

    # Validate the file exists and is MP3, WAV, MP4, or MOV
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' does not exist.")
        return
    if not audio_file.lower().endswith(('.mp3', '.wav', '.mp4', '.mov')):
        print("Error: File must be an MP3, WAV, MP4, or MOV file.")
        return

    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(audio_file))

    # Extract base name without extension for saving files
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    temp_files = []
    transcript = None

    try:
        # Get audio stream info
        audio_info = get_audio_stream_info(audio_file)
        if audio_info:
            print(f"Audio stream: codec={audio_info['codec']}, sample_rate={audio_info['sample_rate']}Hz, channels={audio_info['channels']}, bit_depth={audio_info['bit_depth']}")
        else:
            # Default for YouTube MP4s
            audio_info = {'sample_rate': 44100, 'channels': 2, 'codec': 'aac'}

        optimized_audio_path = os.path.join(input_dir, "optimized_audio.wav")
        temp_files.append(optimized_audio_path)

        # Minimal FFmpeg extraction: No filters, keep stereo, original sample rate to avoid distortion
        cmd = [
            "ffmpeg", "-i", audio_file,
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

        transcript = transcribe_audio(optimized_audio_path, input_dir)

        print("Transcript from audio file:")
        print(transcript)

        # Save the transcript
        transcript_filename = os.path.join(input_dir, f"{base_name}.txt")
        with open(transcript_filename, "w") as file:
            file.write(transcript)
        print(f"Transcript saved to {transcript_filename}")
        if not keep_transcript:
            temp_files.append(transcript_filename)

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr}")
        cleanup(temp_files)
        return
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        cleanup(temp_files)
        return

    # Proceed with summarization if a transcript was obtained
    if transcript:
        # Check if Ollama is running
        if not check_ollama():
            print("\nAborting summarization.")
            cleanup(temp_files)
            return

        print("\nGenerating executive summary and key points...")
        start_time = time.time()
        summary_paragraph, key_points = summarize_with_ollama(transcript)
        end_time = time.time()
        time_taken = end_time - start_time

        print("\n--- Executive Summary ---")
        if summary_paragraph:
            print(summary_paragraph)
        else:
            print("No executive summary found.")

        print("\n--- Key Points ---")
        if key_points:
            for point in key_points:
                print(f"- {point}")
        else:
            print("No key points found.")

        output_filename = os.path.join(input_dir, f"{base_name}_executive_summary.txt")
        with open(output_filename, "w") as f:
            f.write("Executive Summary:\n")
            f.write(summary_paragraph)
            f.write("\n\nKey Points:\n")
            for point in key_points:
                f.write(f"- {point}\n")
        print(f"\nFormatted summary saved to {output_filename}")
        print(f"Time taken for Ollama API call: {time_taken:.2f} seconds")
    else:
        print("No transcript could be generated. Aborting.")

    # Cleanup all temporary files
    cleanup(temp_files)

if __name__ == "__main__":
    main()
