import pymupdf as fitz
from PIL import Image
import io
import pytesseract
import os
from pdf2image import convert_from_path
import numpy as np
import cv2
import requests
import time
import re
import textwrap
from typing import List

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

def chunk_text(text: str, max_words: int = CHUNK_SIZE) -> list[str]:
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
        response.raise_for_status()

        data = response.json()
        raw_output = data["response"].strip()
        print(f"\n--- Raw Ollama Output ---\n{raw_output}\n---\n")

        if is_chunk:
            return raw_output, []

        # Improved line-based parsing
        lines = raw_output.split('\n')
        summary_lines = []
        key_points = []
        in_summary = False
        in_key_points = False

        for line in lines:
            stripped = line.strip()
            lower_stripped = stripped.lower()
            clean_lower = re.sub(r'\*+', '', lower_stripped).strip()
            if clean_lower.startswith('executive summary:'):
                in_summary = True
                in_key_points = False
                continue
            elif clean_lower.startswith('key points:'):
                in_summary = False
                in_key_points = True
                continue
            if in_summary:
                if stripped:
                    summary_lines.append(stripped)
            elif in_key_points:
                if stripped.startswith(('-', '*', '•')) or re.match(r'^\d+\.', stripped):
                    point = re.sub(r'^[-*•\d\.\s]+', '', stripped).strip()
                    if point:
                        key_points.append(point)

        summary_paragraph = '\n'.join(summary_lines).strip()
        summary_paragraph = re.sub(r'^\*+\s*(.*?)\s*\*+$', r'\1', summary_paragraph).strip()

        return summary_paragraph, key_points

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return "", []
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "", []

# Function to get unique filename
def get_unique_filename(filepath: str) -> str:
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path

# Function to preprocess image for better OCR
def preprocess_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to PIL image
    return Image.fromarray(thresh)

# Prompt for summarization at startup
want_summary = input("Do you want a summarization of the document? (y/n): ").lower() == 'y'

# Prompt for the PDF filename
filename = input("Enter PDF filename: ")

# Check if the file exists
if not os.path.exists(filename):
    print(f"Error: File '{filename}' does not exist.")
    exit(1)

input_dir = os.path.dirname(os.path.abspath(filename))
base_name = os.path.splitext(os.path.basename(filename))[0]

full_text = f"Processing PDF: {filename}\n\n"

raw_output_file = os.path.join(input_dir, f"{base_name}_extracted_text.txt")
summary_output_file = os.path.join(input_dir, f"{base_name}_executive_summary.txt")

# Open the PDF document
try:
    doc = fitz.open(filename)
    print(f"Opened PDF: {filename}. Total pages: {len(doc)}")
except Exception as e:
    print(f"Error opening PDF: {e}")
    exit(1)

for page_num in range(len(doc)):
    page = doc[page_num]
    print(f"\nProcessing Page {page_num + 1}")
    full_text += f"Page {page_num + 1}:\n"

    # First, try direct text extraction
    direct_text = page.get_text().strip()
    print(f"Direct text extraction length: {len(direct_text)}")
    if direct_text:
        full_text += f"Direct Text:\n{direct_text}\n"
        print("Direct text extracted successfully.")
    else:
        print("No direct text extracted.")

    # Method 1: Extract images and OCR them (for any embedded images with text)
    image_list = page.get_images(full=True)
    print(f"Number of images found: {len(image_list)}")
    images_found_with_text = False

    for img_index, img in enumerate(image_list):
        print(f"Processing Image {img_index + 1} on Page {page_num + 1}")
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Preprocess image
            processed_image = preprocess_image(image)

            # Perform OCR
            ocr_text = pytesseract.image_to_string(processed_image, config='--psm 6').strip()
            print(f"OCR text length from image {img_index + 1}: {len(ocr_text)}")

            # Append extracted text
            if ocr_text:
                images_found_with_text = True
                full_text += f"Image {img_index + 1}:\n{ocr_text}\n"
            else:
                full_text += f"Image {img_index + 1}: No text detected.\n"
                print("No text detected in image.")
        except Exception as e:
            full_text += f"Image {img_index + 1}: Error extracting text: {e}\n"
            print(f"Error processing image {img_index + 1}: {e}")

    # Method 2: If no direct text and no image text, try rendering whole page with pdf2image and OCR
    if not direct_text and not images_found_with_text:
        print("No text from direct extraction or images. Falling back to full page OCR.")
        try:
            # Convert page to image using pdf2image
            images = convert_from_path(filename, first_page=page_num+1, last_page=page_num+1, dpi=300)
            print(f"Rendered {len(images)} images for page {page_num + 1}")

            for i, image in enumerate(images):
                print(f"Processing rendered image {i + 1} for OCR")
                # Preprocess image
                processed_image = preprocess_image(image)

                # Perform OCR
                ocr_text = pytesseract.image_to_string(processed_image, config='--psm 6').strip()
                print(f"OCR text length from rendered image {i + 1}: {len(ocr_text)}")

                if ocr_text:
                    full_text += f"Rendered Page {page_num + 1}, Image {i + 1}:\n{ocr_text}\n"
                else:
                    full_text += f"Rendered Page {page_num + 1}, Image {i + 1}: No text detected.\n"
                    print("No text detected in rendered image.")
        except Exception as e:
            full_text += f"Rendered Page {page_num + 1}: Error rendering page: {e}\n"
            print(f"Error rendering page {page_num + 1}: {e}")

    full_text += "\n" + "-" * 40 + "\n"  # Separator between pages

# Close the document
doc.close()
print("\nPDF processing complete.")

# Save raw extracted text
raw_output_file = get_unique_filename(raw_output_file)
with open(raw_output_file, "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"Raw extraction complete. Text saved to '{raw_output_file}'.")

summary_paragraph = ""
key_points = []
if want_summary:
    if check_ollama():
        text_to_summarize = full_text
        print("Generating summary...")
        start_time = time.time()
        summary_paragraph, key_points = summarize_with_ollama(text_to_summarize)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for summarization: {time_taken:.2f} seconds")

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

        if summary_paragraph or key_points:
            summary_output_file = get_unique_filename(summary_output_file)
            with open(summary_output_file, "w", encoding="utf-8") as f:
                f.write("Executive Summary:\n")
                f.write(summary_paragraph + "\n\n")
                f.write("Key Points:\n")
                for point in key_points:
                    f.write(f"- {point}\n")
            print(f"Summary saved to '{summary_output_file}'.")
        else:
            print("Failed to generate summary.")
    else:
        print("Ollama not available. Cannot perform summarization.")

# Report locations
print("\nFile locations:")
print(f"Raw extracted file: {raw_output_file}")
if want_summary and (summary_paragraph or key_points):
    print(f"Summary document: {summary_output_file}")
