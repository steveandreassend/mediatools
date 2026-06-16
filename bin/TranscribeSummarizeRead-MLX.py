import os
import sys
import re
import textwrap
import json
import time
import argparse
import yt_dlp
import torch
import functools
import subprocess
import nltk
from typing import List, Tuple
from pydub import AudioSegment
from pydub.effects import normalize
import mlx_whisper
from mlx_lm import load, generate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from xml.etree.ElementTree import ParseError

# --- GUI / MEDIA IMPORTS ---
try:
    from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtCore import QUrl, Qt
except ImportError:
    print("Error: PyQt6 is not installed. Please run 'python3.10 -m pip install PyQt6'.")
    sys.exit(1)

# --- STYLETTS2 STABILITY FIXES ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    from styletts2 import tts
except ImportError:
    print("Error: styletts2 is not installed. Please run 'python3.10 -m pip install styletts2'.")
    tts = None
# ---------------------------------

# Increased limit to take advantage of Qwen2.5's massive context window
DEFAULT_WORD_LIMIT = 50000
CHUNK_SIZE = 50000
LLM_MODEL_ID = "mlx-community/Qwen2.5-14B-Instruct-4bit"

# ==========================================
# AUDIO PLAYER GUI CLASS
# ==========================================
class AudioPlayerWindow(QWidget):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.init_ui()
        self.init_audio()

    def format_time(self, ms):
        """Helper to convert milliseconds to MM:SS or HH:MM:SS format."""
        seconds = (ms // 1000) % 60
        minutes = (ms // (1000 * 60)) % 60
        hours = (ms // (1000 * 60 * 60)) % 24
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"

    def init_ui(self):
        self.setWindowTitle("Summary Player")
        self.setFixedSize(450, 130)

        # Keep window floating on top for easy access
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()

        self.info_label = QLabel("Playing Generated Summary...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.info_label)

        # --- Draggable Progress Bar with Timestamps ---
        slider_layout = QHBoxLayout()

        self.current_time_label = QLabel("00:00")
        slider_layout.addWidget(self.current_time_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        slider_layout.addWidget(self.slider)

        self.total_time_label = QLabel("00:00")
        slider_layout.addWidget(self.total_time_label)

        layout.addLayout(slider_layout)

        # --- Controls ---
        controls = QHBoxLayout()

        self.rewind_btn = QPushButton("⏮ Restart")
        self.rewind_btn.clicked.connect(self.rewind)
        controls.addWidget(self.rewind_btn)

        self.skip_back_btn = QPushButton("⏪ -15s")
        self.skip_back_btn.clicked.connect(self.skip_back)
        controls.addWidget(self.skip_back_btn)

        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.clicked.connect(self.play_pause)
        controls.addWidget(self.play_pause_btn)

        self.skip_fwd_btn = QPushButton("⏩ +15s")
        self.skip_fwd_btn.clicked.connect(self.skip_forward)
        controls.addWidget(self.skip_fwd_btn)

        self.speed_combo = QComboBox()

        # Mapped speeds: The original 1.0x rate is mapped to the "0.8x" UI label
        self.speed_mapping = {
            "0.8x": 1.0,
            "1.0x": 1.25,
            "1.2x": 1.5,
            "1.5x": 1.875,
            "2.0x": 2.5
        }
        self.speed_combo.addItems(list(self.speed_mapping.keys()))
        self.speed_combo.setCurrentText("1.2x")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.speed_combo.setToolTip("Playback Speed")
        controls.addWidget(self.speed_combo)

        layout.addLayout(controls)
        self.setLayout(layout)

    def init_audio(self):
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile(self.filepath))

        # Connect signals for the progress bar
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)

        # Set default mapped speed (1.2x label maps to 1.5 playback rate)
        self.player.setPlaybackRate(self.speed_mapping["1.2x"])

        # Start playing automatically
        self.player.play()

    def position_changed(self, position):
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))

    def set_position(self, position):
        self.player.setPosition(position)
        self.current_time_label.setText(self.format_time(position))

    def play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_pause_btn.setText("▶ Play")
        else:
            self.player.play()
            self.play_pause_btn.setText("⏸ Pause")

    def skip_back(self):
        new_pos = max(0, self.player.position() - 15000)
        self.player.setPosition(new_pos)

    def skip_forward(self):
        new_pos = min(self.player.duration(), self.player.position() + 15000)
        self.player.setPosition(new_pos)

    def rewind(self):
        self.player.setPosition(0)

    def change_speed(self, text):
        speed = self.speed_mapping.get(text, 1.5)
        self.player.setPlaybackRate(speed)

# ==========================================
# CORE SCRIPT LOGIC
# ==========================================

def chunk_text(text: str, max_words: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_with_mlx(text: str, model, tokenizer, is_chunk: bool = False) -> Tuple[str, List[str], str]:
    word_count = len(re.findall(r'\b\w+\b', text))

    if word_count > DEFAULT_WORD_LIMIT and not is_chunk:
        print(f"Video transcript is extremely large ({word_count} words). Splitting into chunks...", flush=True)
        chunks = chunk_text(text, max_words=CHUNK_SIZE)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            print(f"\nSummarizing chunk {i+1}/{len(chunks)}...", flush=True)
            chunk_summary, _, _ = summarize_with_mlx(chunk, model, tokenizer, is_chunk=True)
            chunk_summaries.append(chunk_summary)

        combined_text = "\n\n".join(chunk_summaries)
        print("\nGenerating final summary from chunk summaries...", flush=True)
        return summarize_with_mlx(combined_text, model, tokenizer)

    if is_chunk:
        system_prompt = "You are a helpful assistant. Summarize the provided video transcript chunk in detail."
        user_prompt = textwrap.dedent(f"""\
            Summarize the following video transcript chunk in detail. The summary should be a comprehensive paragraph covering all key aspects.

            ---
            VIDEO TRANSCRIPT CHUNK:
            {text}
            ---

            Please respond with only the summary paragraph.
            """)
    else:
        system_prompt = "You are a helpful assistant. Your task is to extract a detailed executive summary, a comprehensive list of key points, and a conclusion from the provided video transcript."
        user_prompt = textwrap.dedent(f"""\
            Please analyze the following video transcript.

            ---
            VIDEO TRANSCRIPT:
            {text}
            ---

            Respond with a detailed paragraph (200-400 words) for the executive summary, followed by a list of key points, and finally a concluding paragraph. Make the executive summary thorough and insightful, covering main themes and arguments. For key points, provide up to 30 detailed bullet points, each elaborating on important elements. The conclusion should summarize the final thoughts or takeaways. Use the following format strictly:

            Executive Summary:
            [Write the detailed summary paragraph here.]

            Key Points:
            - [List the first key point here, with detail.]
            - [List the second key point here, with detail.]
            - [Continue with up to 30 key points, using bullet points.]

            Conclusions:
            [Write the concluding paragraph here.]
            """)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        print("\nGenerating: \n", flush=True)

        # verbose=True streams the output to the console as it generates
        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=2048,
            verbose=True
        )

        print("\n\n-----------------------------------\n", flush=True)

        if is_chunk:
            return raw_output.strip(), [], ""

        summary_pattern = r"(?:\*\*|###\s*)?Executive Summary:?(?:\*\*)?"
        key_points_pattern = r"(?:\*\*|###\s*)?Key Points:?(?:\*\*)?"
        conclusions_pattern = r"(?:\*\*|###\s*)?Conclusions?:?(?:\*\*)?"

        summary_paragraph = ""
        key_points = []
        conclusion_paragraph = ""

        match_three = re.search(f"(?:^|\n\s*){summary_pattern}(.*?)(?:^|\n\s*){key_points_pattern}(.*?)(?:^|\n\s*){conclusions_pattern}(.*)", raw_output, re.DOTALL | re.IGNORECASE)

        if match_three:
            summary_paragraph = match_three.group(1).strip()
            points_text = match_three.group(2).strip()
            conclusion_paragraph = match_three.group(3).strip()
        else:
            match_two = re.search(f"(?:^|\n\s*){summary_pattern}(.*?)(?:^|\n\s*){key_points_pattern}(.*)", raw_output, re.DOTALL | re.IGNORECASE)
            if match_two:
                summary_paragraph = match_two.group(1).strip()
                points_text = match_two.group(2).strip()
            else:
                summary_match = re.search(f"(?:^|\n\s*){summary_pattern}(.*)", raw_output, re.DOTALL | re.IGNORECASE)
                if summary_match:
                    summary_paragraph = summary_match.group(1).strip()
                points_text = ""

        summary_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', summary_paragraph, flags=re.MULTILINE).strip()

        if points_text:
            points_list = re.findall(r'^\s*(?:-|\*|•|\d+\.)\s*(.*?)$', points_text, re.MULTILINE)
            key_points = [re.sub(r'^\s*\*+\s*(.*?)\s*\*+\s*$', r'\1', point).strip() for point in points_list if point.strip()]

        if conclusion_paragraph:
            conclusion_paragraph = re.sub(r'^\s*\*\*(.*?)\*\*\s*$', r'\1', conclusion_paragraph, flags=re.MULTILINE).strip()

        return summary_paragraph, key_points, conclusion_paragraph

    except Exception as e:
        print(f"An unexpected error occurred during MLX summarization: {e}", flush=True)
        return "", [], ""

def get_transcript(video_url, language='en'):
    video_id = video_url.split("v=")[-1].split("&")[0]
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
        print(f"Error fetching YouTube subtitles: {e}")
        return video_id, None
    except Exception as e:
        print(f"Unexpected error fetching transcript: {e}")
        return video_id, None

def process_and_transcribe_audio(file_path: str) -> str:
    print(f"Processing audio file: {file_path}")
    audio = AudioSegment.from_file(file_path)

    if audio.channels > 1:
        print("Converting stereo to mono by downmixing...")
        audio = audio.set_channels(1)

    print("Normalizing audio volume...")
    audio = normalize(audio)
    audio = audio.set_frame_rate(16000)

    temp_wav = "optimized_temp.wav"
    audio.export(temp_wav, format="wav")

    print("\nTranscribing with local MLX Whisper (large-v3-turbo)...")
    start_time = time.time()

    result = mlx_whisper.transcribe(
        temp_wav,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    )
    transcript = result["text"]

    print(f"Transcription complete in {time.time() - start_time:.2f} seconds.")

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return transcript

def read_summary_aloud(filepath: str, output_wav: str):
    if tts is None:
        print("\nSkipping audio synthesis: styletts2 is not available.")
        return

    print("\nLoading StyleTTS 2 model to read the summary aloud...")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            line = re.sub(r'^([a-zA-Z0-9]+\.|[-*•]+)\s*', '', line)
            sentences = nltk.sent_tokenize(line)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'

                words = sentence.split()
                if len(words) > 60:
                    for i in range(0, len(words), 60):
                        chunk = " ".join(words[i:i+60])
                        if not chunk.endswith(('.', '!', '?')):
                            chunk += '.'
                        cleaned_lines.append(chunk)
                else:
                    cleaned_lines.append(sentence)

        if not cleaned_lines:
            print("Error: The provided text file is empty. Nothing to speak.")
            return

        my_tts = tts.StyleTTS2()

        print(f"Synthesizing audio in {len(cleaned_lines)} segments. This prevents tensor overflow...")

        combined_audio = AudioSegment.empty()
        silence = AudioSegment.silent(duration=400)

        for i, chunk in enumerate(cleaned_lines):
            print(f"  -> Synthesizing segment {i+1}/{len(cleaned_lines)}...", flush=True)
            temp_chunk_wav = f"temp_chunk_{i}.wav"

            my_tts.inference(chunk, output_wav_file=temp_chunk_wav)
            segment = AudioSegment.from_wav(temp_chunk_wav)
            combined_audio += segment + silence

            if os.path.exists(temp_chunk_wav):
                os.remove(temp_chunk_wav)

        print(f"Saving final compiled audio to {output_wav}...")
        combined_audio.export(output_wav, format="wav")
        print(f"Success! The audio has been saved to: {os.path.abspath(output_wav)}")

        print("\nLaunching media player...")
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        abs_output_wav = os.path.abspath(output_wav)
        player_window = AudioPlayerWindow(abs_output_wav)
        player_window.show()

        app.exec()

    except Exception as e:
        print(f"\nFailed to generate or play audio: {e}")

def cleanup(file_list):
    print("\nCleaning up temporary files...")
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)
            print(f" - Removed {f}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribes, summarizes, and reads aloud a YouTube video using local models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        Example usage:
          python transcribeSummarize.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
          python transcribeSummarize.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ -k
          python transcribeSummarize.py # Will prompt for a URL
        ''')
    )
    parser.add_argument("--url", type=str, help="The YouTube video URL to process.")
    parser.add_argument("-k", "--keep-transcript", action="store_true", default=False,
                        help="Retain the raw transcript in a text file.")
    args = parser.parse_args()

    video_url = args.url
    if video_url is None:
        video_url = input("Please enter the YouTube video URL: ")

    keep_transcript = args.keep_transcript
    input_dir = os.getcwd()
    temp_files = []

    video_id, transcript = get_transcript(video_url)

    if transcript is None:
        print("Failed to fetch YouTube subtitles. Falling back to local MLX Whisper transcription...")
        try:
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

            if not os.path.exists(audio_path):
                raise FileNotFoundError("Downloaded audio file not found.")

            transcript = process_and_transcribe_audio(audio_path)

        except Exception as e:
            print(f"Failed to download or transcribe audio: {e}")
            cleanup(temp_files)
            return

    if not transcript or not transcript.strip():
        print("No transcript could be generated. Aborting.")
        cleanup(temp_files)
        return

    transcript_filename = os.path.join(input_dir, f"{video_id}_transcript.txt")
    with open(transcript_filename, "w", encoding="utf-8") as file:
        file.write(transcript)

    print(f"Transcript saved to {transcript_filename}")
    if not keep_transcript:
        temp_files.append(transcript_filename)

    # --- Initialize MLX-LM ---
    print(f"\nLoading language model ({LLM_MODEL_ID}) into unified memory...")
    try:
        model, tokenizer = load(LLM_MODEL_ID)
    except Exception as e:
        print(f"\nFailed to load the language model: {e}")
        cleanup(temp_files)
        return

    print("\nGenerating executive summary, key points, and conclusions...")
    start_time = time.time()
    summary_paragraph, key_points, conclusion_paragraph = summarize_with_mlx(transcript, model, tokenizer)
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

    print("\n--- Conclusions ---")
    if conclusion_paragraph:
        print(conclusion_paragraph)
    else:
        print("No conclusions found.")

    output_filename = os.path.join(input_dir, f"{video_id}_executive_summary.txt")
    with open(output_filename, "w", encoding="utf-8") as f:
        if summary_paragraph:
            f.write("Executive Summary:\n")
            f.write(summary_paragraph + "\n\n")
        if key_points:
            f.write("Key Points:\n")
            for point in key_points:
                f.write(f"- {point}\n")
            f.write("\n")
        if conclusion_paragraph:
            f.write("Conclusions:\n")
            f.write(conclusion_paragraph + "\n\n")

    print(f"\nFormatted summary saved to {output_filename}")
    print(f"Time taken for MLX generation: {time_taken:.2f} seconds")

    audio_output_filename = os.path.join(input_dir, f"{video_id}_summary_audio.wav")
    read_summary_aloud(output_filename, audio_output_filename)

    cleanup(temp_files)

if __name__ == "__main__":
    main()
