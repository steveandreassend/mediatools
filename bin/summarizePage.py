import os
import re
import argparse
import textwrap
import json
import requests
import time
from bs4 import BeautifulSoup
from langdetect import detect
from googletrans import Translator
from playwright.sync_api import sync_playwright
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

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

def extract_web_text_and_metadata(url: str) -> (str, int, int):
    """Extracts text, sets page count to 1, and word count from a webpage using Playwright."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            # Navigate to the URL with retries
            max_retries = 3
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    print(f"Loading page {url} (attempt {retry_count + 1})")
                    page.goto(url, timeout=90000)  # Increased timeout to 90s
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"Failed to load page {url} after {max_retries + 1} attempts: {str(e)}")
                        browser.close()
                        return None, 0, 0
                    print(f"Retrying page load for {url}: {str(e)}")
                    page.wait_for_timeout(5000)

            # Wait for network idle
            try:
                page.wait_for_load_state('networkidle', timeout=90000)
                print(f"Network idle for {url}")
            except Exception as e:
                print(f"Network idle timeout for {url}: {str(e)}. Proceeding with extraction.")

            # Handle consent buttons
            consent_selectors = [
                "button:contains('Accept'), button:contains('Confirm'), button:contains('Agree'), button:contains('Allow'), "
                "button:contains('OK'), button:contains('Close'), button:contains('Akkoord'), button:contains('Toestaan'), "
                "button:contains('Accepteren'), button:contains('Ik ga akkoord'), button[class*='accept'], "
                "button[class*='confirm'], button[class*='cookie'], button[class*='consent'], button[id*='accept'], "
                "button[id*='consent'], button[title*='accept'], div[id*='consent'] button, div[class*='modal'] button, "
                "button[class*='btn--primary'], button[class*='btn--accept'], div[class*='privacy-gate'] button, "
                "button[data-testid*='accept'], button[data-cy*='accept'], button[class*='dp-btn--primary'], "
                "button[class*='privacy-gate__button'], button[class*='js-accept-cookies'], "
                "button[data-action*='accept'], button:contains('Accepteer cookies'), "
                "div[class*='privacy-gate'] button:contains('Akkoord'), button[class*='dpg-btn--primary']"
            ]
            for selector in consent_selectors:
                try:
                    consent_button = page.query_selector(selector)
                    if consent_button:
                        consent_button.click()
                        print(f"Clicked consent button for {url} with selector '{selector}'")
                        page.wait_for_timeout(7000)  # Increased to 7s
                        break
                except Exception:
                    continue
            else:
                print(f"No consent button found for {url} with any selector")

            # Check for iframes
            iframes = page.query_selector_all('iframe')
            for iframe in iframes:
                try:
                    frame = iframe.content_frame()
                    if frame:
                        for selector in consent_selectors:
                            consent_button = frame.query_selector(selector)
                            if consent_button:
                                consent_button.click()
                                print(f"Clicked consent button in iframe for {url} with selector '{selector}'")
                                page.wait_for_timeout(7000)
                                break
                except Exception:
                    continue

            # Check for CAPTCHA
            try:
                captcha_elements = page.query_selector_all('div[id*="captcha"], div[class*="captcha"]')
                if captcha_elements:
                    print(f"Warning: Possible CAPTCHA detected for {url}. Manual intervention may be required.")
            except Exception as e:
                print(f"Failed to check for CAPTCHA on {url}: {str(e)}")

            # Scroll to load dynamic content
            try:
                for _ in range(10):  # Increased to 10 scrolls
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    print(f"Scrolled page {url} to load dynamic content")
                    page.wait_for_timeout(3000)  # Increased to 3s
            except Exception as e:
                print(f"Failed to scroll page {url}: {str(e)}")

            # Extract page content
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            browser.close()

            # Remove cookie/consent banners
            for element in soup.find_all(['div', 'section', 'aside'], class_=re.compile(r'cookie|consent|gdpr|banner|prompt|overlay|modal|popup|privacy-gate', re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(r'cookie|consent|gdpr|banner|prompt|overlay|modal|popup|privacy-gate', re.I)):
                element.decompose()

            # Extract cleaned text
            text = soup.get_text(separator='\n', strip=True)
            word_count = len(re.findall(r'\b\w+\b', text))
            num_pages = 1  # Webpages don't have pages, but set to 1 for compatibility

            # Warn if text is too short
            if word_count < 50:
                print(f"Warning: Extracted text is very short ({word_count} words). The webpage may be behind a privacy gate, paywall, or login screen.")

            return text, num_pages, word_count
    except Exception as e:
        print(f"Error extracting webpage text and metadata: {e}")
        return None, 0, 0

def detect_and_translate_text(text: str) -> str:
    """Detects the language of the text and translates to English if necessary, returning the text to summarize."""
    if not text or not text.strip():
        print("No text to detect or translate.")
        return text

    try:
        # Detect language using up to 500 chars for efficiency
        lang = detect(text[:500])
        print(f"\nDetected language: {lang}")

        if lang == 'en':
            print("Text is already in English. No translation needed.")
            return text
        else:
            print("Text is not in English. Translating to English...")
            translator = Translator()
            # Translate in chunks to avoid API limits
            max_chunk_size = 5000  # Google Translate API limit per request
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_text = ""
            for i, chunk in enumerate(chunks):
                try:
                    translated_chunk = translator.translate(chunk, dest='en').text
                    translated_text += translated_chunk + "\n"
                except Exception as e:
                    print(f"Translation failed for chunk {i+1}: {e}")
                    translated_text += chunk + "\n"
            return translated_text

    except Exception as e:
        print(f"Error during language detection or translation: {e}")
        return text

def chunk_text(text: str, max_words: int = CHUNK_SIZE) -> List[str]:
    """Splits text into chunks of a maximum word count."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_with_ollama(text: str, is_chunk: bool = False) -> (str, List[str]):
    """
    Sends text to Ollama for summarization. Handles large text by chunking.

    If is_chunk is True, it returns only the summary paragraph for a single chunk.
    Otherwise, it returns the final executive summary and key points.
    Always summarizes in English, regardless of the original language.
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
            Summarize the following document chunk in detail, in English, regardless of the original language. The summary should be a comprehensive paragraph covering all key aspects.

            ---
            DOCUMENT CHUNK:
            {text}
            ---

            Please respond with only the summary paragraph.
            """)
    else:
        # Final summary prompt
        prompt = textwrap.dedent(f"""\
            You are a helpful assistant. Your task is to extract a detailed executive summary and a comprehensive list of key points from the provided document text, in English, regardless of the original language.

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
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        raw_output = data["response"].strip()
        print(f"\n--- Raw Ollama Output for Parsing ---\n{raw_output}\n-----------------------------------\n")

        # If summarizing a chunk, just return the raw output.
        if is_chunk:
            return raw_output, []

        # Parse the raw output for the final summary and key points.
        summary_paragraph = ""
        key_points = []

        # Use regex to find and split the sections, allowing for optional markdown bold
        match = re.search(r"(?:^|\n\s*)(?:\*\*)?\s*Executive Summary\s*:\s*(?:\*\*)?\s*(.*?)(?:^|\n\s*)(?:\*\*)?\s*Key Points\s*:\s*(?:\*\*)?\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)

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
            summary_match = re.search(r"(?:^|\n\s*)(?:\*\*)?\s*Executive Summary\s*:\s*(?:\*\*)?\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary_paragraph = summary_match.group(1).strip()
                summary_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', summary_paragraph, flags=re.MULTILINE).strip()

        return summary_paragraph, key_points

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return "", []
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "", []

def main():
    parser = argparse.ArgumentParser(description="Generate an executive summary from a webpage using the local Llama3 model via a direct Ollama API call.")
    parser.add_argument('--url', help="URL of the webpage", default=None)
    args = parser.parse_args()

    # Prompt for URL if not provided
    url = args.url
    if not url:
        url = input("Please enter the URL of the webpage to summarize (e.g., https://example.com): ").strip()

    if not url.startswith('http'):
        print(f"Invalid URL: {url}. It should start with http:// or https://.")
        return

    if not check_ollama():
        print("\nAborting summarization.")
        return

    # Generate a filename from the URL
    parsed_url = urlparse(url)
    filename = (parsed_url.netloc + parsed_url.path.strip('/').replace('/', '_')) or 'webpage'
    response = requests.get(url)  # Quick request to get size
    file_size_mb = len(response.content) / (1024 * 1024)
    text, num_pages, num_words = extract_web_text_and_metadata(url)
    if text is None or not text.strip():
        print("Failed to extract meaningful text from webpage. Aborting.")
        return

    # Detect and translate text if necessary
    text = detect_and_translate_text(text)

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
