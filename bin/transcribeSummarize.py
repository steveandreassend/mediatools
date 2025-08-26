import os
import re
import textwrap
import json
import requests
import time
import math
import argparse
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from xml.etree.ElementTree import ParseError
from pydub import AudioSegment
import speech_recognition as sr
import yt_dlp
from typing import List, Dict, Any

# Google Transcription API
# Milliseconds of audio to process per request to limit timeout errors
TRANSCRIPTION_CHUNK_SIZE = 30000

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
            Summarize the following document chunk. The summary should be a single paragraph.

            ---
            DOCUMENT CHUNK:
            {text}
            ---

            Please respond with only the summary paragraph.
            """)
    else:
        # Final summary prompt
        prompt = textwrap.dedent(f"""\
            You are a helpful assistant. Your task is to extract a concise executive summary and a list of key points from the provided document text.

            ---
            DOCUMENT TEXT:
            {text}
            ---

            Please respond with a single paragraph for the executive summary, followed by a list of key points. Use the following format:

            Executive Summary:
            [Write the concise summary paragraph here.]

            Key Points:
            - [List the first key point here.]
            - [List the second key point here.]
            - [Continue with up to 20 key points, using bullet points.]
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

def get_transcript(video_url, language='en'):
    """Fetches the transcript from a YouTube video."""
    video_id = video_url.split("v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcript_list.find_transcript([language])
            transcript_data = transcript.fetch()
            return video_id, "\n".join([entry.text for entry in transcript_data])
        except NoTranscriptFound:
            print(f"No transcript in {language}. Attempting to find a generated one.")
            transcript = transcript_list.find_generated_transcript([language])
            transcript_data = transcript.fetch()
            return video_id, "\n".join([entry.text for entry in transcript_data])
        except ParseError:
            print("Parse error while fetching transcript. The transcript may be empty or invalid.")
            return video_id, None
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"Error: {e}")
        return video_id, None
    except Exception as e:
        print(f"Unexpected error fetching transcript: {e}")
        return video_id, None

def transcribe_audio(audio_path):
    print(f"Transcribes an audio file using Google's Web Speech API", flush=True)
    audio = AudioSegment.from_wav(audio_path)
    recognizer = sr.Recognizer()
    transcript = ""
    print(f"Chunk length: {TRANSCRIPTION_CHUNK_SIZE}", flush=True)
    iterations = math.ceil(len(audio) / TRANSCRIPTION_CHUNK_SIZE)
    print(f"Number of chunks to process: {iterations}", flush=True)
    for i in range(0, len(audio), TRANSCRIPTION_CHUNK_SIZE):
        print(f"Processing chunk {math.ceil(i/TRANSCRIPTION_CHUNK_SIZE + 1)}... ")
        chunk = audio[i:i + TRANSCRIPTION_CHUNK_SIZE]
        temp_chunk_path = "temp_chunk.wav"
        chunk.export(temp_chunk_path, format="wav")
        with sr.AudioFile(temp_chunk_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            transcript += text + " "
        except sr.UnknownValueError:
            print(f"Could not understand audio chunk starting at {i/1000} seconds", flush=True)
        except sr.RequestError as e:
            print(f"Could not request results for chunk starting at {i/1000} seconds: {e}", flush=True)
    if os.path.exists("temp_chunk.wav"):
        os.remove("temp_chunk.wav")
    return transcript.strip()

def cleanup(file_list):
    """Removes a list of files if they exist."""
    print("\nCleaning up temporary files...")
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)
            print(f" - Removed {f}")

def main():
    """Main function to orchestrate the transcription and summarization process."""
    parser = argparse.ArgumentParser(
        description="Transcribes and summarizes a YouTube video using Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        Example usage:
          python transcribeSummarize.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
          python transcribeSummarize.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ -k
          python transcribeSummarize.py # Will prompt for a URL
        ''')
    )
    # The --url argument is no longer required.
    parser.add_argument("--url", type=str,
                        help="The YouTube video URL to process.")
    parser.add_argument("-k", "--keep-transcript", action="store_true", default=False,
                        help="Retain the raw transcript in a text file.")
    args = parser.parse_args()

    video_url = args.url
    # If the URL is not provided as a command-line argument, prompt the user for it.
    if video_url is None:
        video_url = input("Please enter the YouTube video URL: ")

    keep_transcript = args.keep_transcript

    video_id, transcript = get_transcript(video_url)

    temp_files = []

    # If transcript is None, try audio transcription
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
            optimized_audio_path = "optimized_audio.wav"
            temp_files.extend([audio_path, optimized_audio_path])

            # Optimize audio for transcription
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(44100).set_channels(1)
            audio.export(optimized_audio_path, format="wav")

            transcript = transcribe_audio(optimized_audio_path)

            print("Transcript from audio file:")
            print(transcript)

            # Save the transcript if the keep_transcript flag is set
            transcript_filename = f"{video_id}.txt"
            with open(transcript_filename, "w") as file:
                file.write(transcript)
            print(f"Transcript saved to {transcript_filename}")
            if not keep_transcript:
                temp_files.append(transcript_filename)

        except Exception as e:
            print(f"Failed to download or transcribe audio: {e}")
            cleanup(temp_files)
            return
    else:
        print("Transcript from YouTube subtitles:")
        print(transcript)
        transcript_filename = f"{video_id}.txt"
        with open(transcript_filename, "w") as file:
            file.write(transcript)
        print(f"Transcript saved to {transcript_filename}")
        if not keep_transcript:
            temp_files.append(transcript_filename)


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

        output_filename = f"{video_id}_executive_summary.txt"
        with open(output_filename, "w") as f:
            f.write("Executive Summary:\n")
            f.write(summary_paragraph)
            f.write("\n\nKey Points:\n")
            for point in key_points:
                f.write(f"- {point}\n")
        print(f"\nFormatted summary saved to {output_filename}")
        print(f"Time taken for Ollama API call: {time_taken:.2f} seconds")
    else:
        print("No transcript found or could be generated. Aborting.")

    # Cleanup all temporary files
    cleanup(temp_files)

if __name__ == "__main__":
    main()
