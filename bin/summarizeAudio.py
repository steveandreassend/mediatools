import os
import argparse
import time
import requests
import json
import re
import textwrap
from typing import List, Tuple
from pydub import AudioSegment
from pydub.effects import normalize
import mlx_whisper

# Constants for chunking long transcripts
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
        print("Could not connect to Ollama. Is the server running (ollama serve)?")
        return False

def chunk_text(text: str, max_words: int) -> List[str]:
    """Splits a long text into smaller chunks by word count."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_with_ollama(text: str, is_chunk: bool = False) -> Tuple[str, List[str]]:
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

    # Enable streaming and increase the context window
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": 8192 # Safely handles the 2000-word CHUNK_SIZE
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=300)
        response.raise_for_status()

        raw_output = ""
        print("Generating: ", end="", flush=True)

        # Iterate over the stream to print progress in real-time
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                chunk_text_resp = body.get("response", "")
                raw_output += chunk_text_resp
                print(chunk_text_resp, end="", flush=True)

        print("\n\n-----------------------------------\n", flush=True)

        if is_chunk:
            return raw_output.strip(), []

        # Robust parsing logic to extract sections
        summary_start_marker = "Executive Summary:"
        key_points_start_marker = "Key Points:"

        summary_paragraph = ""
        key_points = []

        match = re.search(f"(?:^|\n\s*){re.escape(summary_start_marker)}(.*?)(?:^|\n\s*){re.escape(key_points_start_marker)}(.*)", raw_output, re.DOTALL | re.IGNORECASE)

        if match:
            summary_paragraph = match.group(1).strip()
            summary_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', summary_paragraph, flags=re.MULTILINE).strip()

            points_text = match.group(2).strip()
            points_list = re.findall(r'^\s*(?:-|\*|•|\d+\.)\s*(.*?)$', points_text, re.MULTILINE)
            key_points = [re.sub(r'^\s*\*+\s*(.*?)\s*\*+\s*$', r'\1', point).strip() for point in points_list if point.strip()]
        else:
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

def process_audio(file_path: str) -> str:
    """Pre-processes audio for clarity and transcribes it using MLX Whisper."""
    print(f"Processing audio file: {file_path}")

    # Load audio
    audio = AudioSegment.from_file(file_path)

    # Downmix stereo to mono safely so no channels are dropped
    if audio.channels > 1:
        print("Converting stereo to mono by downmixing...")
        audio = audio.set_channels(1)

    # Normalize audio volume so speech is loud and clear
    print("Normalizing audio volume...")
    audio = normalize(audio)

    # Set to 16kHz for optimal Whisper compatibility
    audio = audio.set_frame_rate(16000)

    # Export optimized temporary wav for transcription
    temp_wav = "optimized_temp.wav"
    audio.export(temp_wav, format="wav")

    # Transcribe using mlx-whisper (Optimized for Apple Silicon GPU)
    print("\nTranscribing with local MLX Whisper (large-v3-turbo)...")
    start_time = time.time()

    result = mlx_whisper.transcribe(
        temp_wav,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    )
    transcript = result["text"]

    print(f"Transcription complete in {time.time() - start_time:.2f} seconds.")

    # Cleanup temp file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return transcript

def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize an audio/video file entirely locally.")
    parser.add_argument("input_file", help="Path to the mp4, mp3, or wav file")
    args = parser.parse_args()

    input_file = args.input_file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    input_dir = os.path.dirname(os.path.abspath(input_file))

    if not os.path.exists(input_file):
        print("Error: File not found.")
        return

    # 1. Transcribe the file
    transcript = process_audio(input_file)

    if not transcript or not transcript.strip():
        print("No transcript could be generated. Aborting.")
        return

    print("\n--- Full Transcript ---")
    print(transcript)

    # 2. Summarize the transcript
    if not check_ollama():
        print("\nAborting summarization because Ollama is not ready.")
        return

    start_time = time.time()
    summary_paragraph, key_points = summarize_with_ollama(transcript)
    time_taken = time.time() - start_time

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

    # 3. Save output
    output_filename = os.path.join(input_dir, f"{base_name}_executive_summary.txt")
    with open(output_filename, "w", encoding="utf-8") as f:
        if summary_paragraph:
            f.write("Executive Summary:\n")
            f.write(summary_paragraph + "\n\n")
        if key_points:
            f.write("Key Points:\n")
            for point in key_points:
                f.write(f"- {point}\n")

    print(f"\nFormatted summary saved to {output_filename}")
    print(f"Time taken for Ollama API call: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
