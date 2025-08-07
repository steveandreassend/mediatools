import os
import re
import argparse
import textwrap
import json
import requests
import time
from pypdf import PdfReader
from typing import List, Dict, Any

# A default word limit to trigger chunking.
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

def extract_pdf_text_and_metadata(pdf_path: str) -> (str, int, int):
    """Extracts text, page count, and word count from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        word_count = len(re.findall(r'\b\w+\b', text))
        return text, num_pages, word_count
    except Exception as e:
        print(f"Error extracting PDF text and metadata: {e}")
        return None, 0, 0

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
        print(f"Document is large ({word_count} words). Splitting into chunks...")
        chunks = chunk_text(text, max_words=CHUNK_SIZE)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            chunk_summary, _ = summarize_with_ollama(chunk, is_chunk=True)
            chunk_summaries.append(chunk_summary)

        combined_text = "\n\n".join(chunk_summaries)
        print("Generating final summary from chunk summaries...")
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
        print(f"\n--- Raw Ollama Output for Parsing ---\n{raw_output}\n-----------------------------------\n")

        # If summarizing a chunk, just return the raw output.
        if is_chunk:
            return raw_output, []

        # Parse the raw output for the final summary and key points.
        summary_paragraph = ""
        key_points = []

        # More robust regex to find the summary and key points
        summary_match = re.search(r"Executive Summary:?\s*(.*?)(?:\n\s*Key Points:?|\Z)", raw_output, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary_paragraph = summary_match.group(1).strip()

        key_points_match = re.search(r"Key Points:?\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
        if key_points_match:
            points_text = key_points_match.group(1).strip()
            # Use a regex to find all bulleted or numbered list items
            points_list = re.findall(r'^(?:-|\d+\.)\s*(.*?)$', points_text, re.MULTILINE)
            key_points = [point.strip() for point in points_list if point.strip()]

        return summary_paragraph, key_points

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return "", []
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "", []

def main():
    parser = argparse.ArgumentParser(description="Generate an executive summary from a PDF document using the local Llama3 model via a direct Ollama API call.")
    parser.add_argument('pdf_path', help="Path to the PDF document")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    if not check_ollama():
        print("\nAborting summarization.")
        return

    filename = os.path.basename(pdf_path)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    text, num_pages, num_words = extract_pdf_text_and_metadata(pdf_path)
    if text is None or not text.strip():
        print("Failed to extract meaningful text from PDF. Aborting.")
        return

    print(f"\nDocument word count: {num_words}")
    print("\nRunning Ollama summarization...")
    start_time = time.time()
    summary_paragraph, key_points = summarize_with_ollama(text)
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

    output_filename = f"{filename}_executive_summary.txt"
    with open(output_filename, "w") as f:
        f.write("Executive Summary:\n")
        f.write(summary_paragraph)
        f.write("\n\nKey Points:\n")
        for point in key_points:
            f.write(f"- {point}\n")
    print(f"\nFormatted summary saved to {output_filename}")

    print(f"\nFilename: {filename}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Number of pages: {num_pages}")
    print(f"Number of words: {num_words}")
    print(f"Time taken for Ollama API call: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
