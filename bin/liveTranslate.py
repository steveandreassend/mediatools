import os
import sys
import re
import json
import time
import queue
import threading
import textwrap
import tempfile
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import sounddevice as sd
import mlx_whisper
from mlx_lm import load, generate

# --- CONFIGURATION ---
TRANSLATION_MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SUMMARY_MODEL_ID = "mlx-community/Qwen2.5-14B-Instruct-4bit"
SAMPLE_RATE = 16000
PORT = 8080

LANG_MAP = {
    "fr": "FRENCH",
    "es": "SPANISH",
    "de": "GERMAN",
    "it": "ITALIAN",
    "nl": "DUTCH",
    "pt": "PORTUGUESE"
}

# --- GLOBAL STATE MANAGEMENT ---
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_segments = []
        self.summary_output = ""
        self.is_running = True
        self.is_listening = True
        self.chosen_language = "auto"

        # Dual Model State
        self.trans_model = None
        self.trans_tokenizer = None
        self.summary_model = None
        self.summary_tokenizer = None

state = State()
audio_queue = queue.Queue()

# --- AUDIO RECORDING CALLBACK ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio Status Warning: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

# --- HALLUCINATION FILTER ---
def is_hallucination(text: str) -> bool:
    if not text:
        return True
    if len(text) > 50 and " " not in text:
        return True
    words = [w.strip(".,!?").lower() for w in text.split()]
    if len(words) > 4 and len(set(words)) == 1:
        return True
    lower_text = text.lower()
    if "sous-titrage" in lower_text or "amara.org" in lower_text:
        return True
    if "hello, how are you" in lower_text or "i am doing well" in lower_text:
        return True
    stripped = lower_text.strip(".,!? ")
    if stripped in ["thank you", "merci", "hello", "bye"]:
        return True
    return False

# --- LLM LAZY LOADERS ---
def load_translation_model():
    with state.lock:
        if state.trans_model is None or state.trans_tokenizer is None:
            print(f"\n[System] Loading lightweight real-time model: {TRANSLATION_MODEL_ID}...")
            state.trans_model, state.trans_tokenizer = load(TRANSLATION_MODEL_ID)
        return state.trans_model, state.trans_tokenizer

def load_summary_model():
    with state.lock:
        if state.summary_model is None or state.summary_tokenizer is None:
            print(f"\n[System] Loading heavy summary model: {SUMMARY_MODEL_ID}... (This may take a moment)")
            state.summary_model, state.summary_tokenizer = load(SUMMARY_MODEL_ID)
        return state.summary_model, state.summary_tokenizer

# --- ENGLISH-TO-FOREIGN LLM TRANSLATOR (3B MODEL) ---
def translate_to_foreign(text: str, target_lang: str, max_tokens: int = 256) -> str:
    """Translates text. If text is already in the target language, it cleans and returns it."""
    model, tokenizer = load_translation_model()
    system_prompt = f"You are a professional interpreter. Translate the following text into {target_lang}. If the text is already in {target_lang}, simply output the original text. Output ONLY the exact translation. Do not add quotes, UI brackets, or conversational filler."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        raw_output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return raw_output.strip()
    except Exception as e:
        return f"[Translation Error: {e}]"

# --- TTS GENERATOR (Native macOS Voice engine - Forced WAVE format) ---
def generate_tts(text: str, lang: str, output_filepath: str):
    """Uses macOS native 'say' command for reliable, multi-lingual TTS."""
    voice_map = {
        "en": "Alex",
        "fr": "Thomas",
        "es": "Jorge",
        "de": "Anna",
        "it": "Luca",
        "nl": "Xander",
        "pt": "Joana"
    }

    voice = voice_map.get(lang, "Alex")
    clean_text = text.replace('"', '\\"').replace('\n', '. ')

    # Explicitly force a standard RIFF WAV container and 16-bit PCM data
    cmd = [
        'say',
        '-v', voice,
        clean_text,
        '-o', output_filepath,
        '--file-format=WAVE',
        '--data-format=LEI16@44100'
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"TTS Generation failed: {e}")

# --- LIVE TRANSLATION BACKGROUND WORKER ---
def transcription_worker():
    print("Initializing Whisper transcription loop with Voice Activity Detection...")

    # Pre-load the translation model so the first translation is instant
    load_translation_model()

    eval_samples = int(SAMPLE_RATE * 0.5)
    audio_buffer = np.zeros((0, 1), dtype=np.float32)

    speech_buffer = np.zeros((0, 1), dtype=np.float32)
    is_speaking = False
    silence_time = 0.0

    while state.is_running:
        try:
            try:
                data = audio_queue.get(timeout=0.1)
                audio_buffer = np.vstack((audio_buffer, data))
            except queue.Empty:
                pass

            with state.lock:
                listening = state.is_listening

            if not listening:
                audio_buffer = np.zeros((0, 1), dtype=np.float32)
                speech_buffer = np.zeros((0, 1), dtype=np.float32)
                is_speaking = False
                continue

            if len(audio_buffer) >= eval_samples:
                chunk = audio_buffer[:eval_samples]
                audio_buffer = audio_buffer[eval_samples:]

                rms = np.sqrt(np.mean(chunk**2))

                if rms >= 0.015:
                    is_speaking = True
                    silence_time = 0.0
                    speech_buffer = np.vstack((speech_buffer, chunk))
                elif is_speaking:
                    silence_time += 0.5
                    speech_buffer = np.vstack((speech_buffer, chunk))

                if is_speaking and (silence_time >= 1.5 or len(speech_buffer) >= SAMPLE_RATE * 15.0):
                    audio_to_process = speech_buffer.flatten()
                    speech_buffer = np.zeros((0, 1), dtype=np.float32)
                    is_speaking = False
                    silence_time = 0.0

                    with state.lock:
                        lang = state.chosen_language

                    kwargs = {
                        "task": "translate",
                        "initial_prompt": "Speaker (English): Hello, how are you? I am doing well."
                    }

                    result = mlx_whisper.transcribe(audio_to_process, path_or_hf_repo="mlx-community/whisper-large-v2-mlx", **kwargs)
                    text = result.get("text", "").strip()
                    detected_lang = result.get("language", "unknown")

                    if text and not is_hallucination(text):
                        if detected_lang == "en":
                            if lang != "auto":
                                lang_name = LANG_MAP.get(lang, "UNKNOWN")
                                print(f"English detected. Generating {lang_name} translation via LLM...")
                                translated = translate_to_foreign(text, lang_name.capitalize())
                                display_text = f"[EN] {text}\n[{lang_name} TRANSLATION]: {translated}"
                            else:
                                display_text = f"[EN] {text}"
                        else:
                            display_text = f"[{detected_lang.upper()} -> EN] {text}"

                        with state.lock:
                            state.pending_segments.append(display_text.strip())
                            print(display_text.strip())

        except Exception as e:
            print(f"Error in transcription worker: {e}", file=sys.stderr)

# --- MLX CORE SUMMARIZATION LOGIC (14B MODEL) ---
def summarize_transcript(text: str) -> str:
    model, tokenizer = load_summary_model()
    system_prompt = "You are a helpful assistant tasking with extracting structural summaries, core takeaways, and clear action steps from transcripts."
    user_prompt = textwrap.dedent(f"""\
        Please analyze the following live translated text transcript.
        ---
        TRANSCRIPT:
        {text}
        ---
        Respond with a comprehensive Executive Summary paragraph, followed by a list of Key Points, a Concluding paragraph, and finally an Action Items list. Use the following format structure strictly:

        Executive Summary:
        [Write the detailed summary paragraph here.]

        Key Points:
        - [List key points here with detail]

        Conclusions:
        [Write the concluding paragraph here.]

        Action Items:
        - [List actionable tasks and follow-ups derived from the text]
        """)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        print("\nGenerating AI Synthesis summary layout...", flush=True)
        raw_output = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=False)
        return raw_output.strip()
    except Exception as e:
        return f"An unexpected error occurred during MLX summarization: {e}"

# --- NATIVE HTTP SERVER INFRASTRUCTURE ---
class WebServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif self.path == "/api/data":
            with state.lock:
                new_segments = list(state.pending_segments)
                state.pending_segments.clear()

                response_data = {
                    "new_segments": new_segments,
                    "summary": state.summary_output,
                    "is_listening": state.is_listening
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        elif self.path.startswith("/audio/"):
            filename = self.path.split("/")[-1]
            filepath = os.path.join(tempfile.gettempdir(), filename)
            if os.path.exists(filepath):
                self.send_response(200)
                # Ensure standard audio/wav is served
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File Not Found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        if self.path == "/api/set_language":
            data = json.loads(post_data)
            with state.lock:
                state.chosen_language = data.get("language", "auto")
            self.send_response(200)
            self.end_headers()

        elif self.path == "/api/toggle_listen":
            with state.lock:
                state.is_listening = not state.is_listening
            self.send_response(200)
            self.end_headers()

        elif self.path == "/api/summarize":
            data = json.loads(post_data)
            current_text = data.get("text", "")
            if not current_text.strip():
                summary_result = "No text transcript available to synthesize summaries."
            else:
                summary_result = summarize_transcript(current_text)

            with state.lock:
                state.summary_output = summary_result
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode("utf-8"))

        elif self.path == "/api/tts":
            data = json.loads(post_data)
            raw_text = data.get("text", "")
            lang = data.get("language", "en")
            target = data.get("target", "transcript")

            # Extend lang map to handle explicit English selections
            extended_lang_map = dict(LANG_MAP)
            extended_lang_map["en"] = "ENGLISH"
            lang_name = extended_lang_map.get(lang, "ENGLISH").capitalize()

            # Pass text through the LLM to translate or strip out transcript UI tags
            print(f"\n[TTS] Pre-processing text for native {lang_name} pronunciation...")
            clean_text = translate_to_foreign(raw_text, lang_name, max_tokens=1024)

            # Fallback in case of generation error
            if "[Translation Error" in clean_text:
                clean_text = raw_text

            # Standard .wav extension
            filename = f"tts_{target}_{int(time.time())}.wav"
            filepath = os.path.join(tempfile.gettempdir(), filename)

            generate_tts(clean_text, lang, filepath)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"audio_url": f"/audio/{filename}"}).encode("utf-8"))

        elif self.path == "/api/clear":
            with state.lock:
                state.pending_segments.clear()
                state.summary_output = ""
            self.send_response(200)
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

# --- HTML FRONTEND INTERFACE ---
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live Bilingual Translation & Synthesis</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background-color: #f5f5f7; color: #1d1d1f; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        h1 { font-size: 24px; margin-bottom: 20px; text-align: center; }
        .controls { display: flex; gap: 12px; margin-bottom: 15px; align-items: center; flex-wrap: wrap; }
        .tts-row { background-color: #f0f0f5; padding: 12px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #d2d2d7; display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
        select, button { padding: 8px 14px; font-size: 13px; border-radius: 8px; border: 1px solid #d2d2d7; background-color: white; cursor: pointer; transition: all 0.2s; }
        button:hover { background-color: #f5f5f7; }
        button.primary { background-color: #0071e3; color: white; border: none; }
        button.primary:hover { background-color: #0077ed; }
        button.stop-btn { background-color: #ff3b30; color: white; border: none; font-weight: bold; }
        button.stop-btn:hover { background-color: #d70015; }
        button.start-btn { background-color: #34c759; color: white; border: none; font-weight: bold; }
        button.start-btn:hover { background-color: #248a3d; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .box-title { font-weight: 600; margin-top: 20px; margin-bottom: 8px; font-size: 16px; }
        textarea { width: 100%; height: 260px; padding: 12px; border-radius: 8px; border: 1px solid #d2d2d7; font-family: inherit; font-size: 14px; resize: vertical; box-sizing: border-box; background-color: #ffffff; transition: border 0.2s; }
        textarea:focus { outline: none; border: 2px solid #0071e3; }
        .summary-box { background-color: #f4f4f6; white-space: pre-wrap; border-radius: 8px; border: 1px solid #d2d2d7; padding: 15px; min-height: 150px; font-size: 14px; line-height: 1.5; }
        .tts-section { display: flex; align-items: center; gap: 8px; padding-right: 15px; border-right: 1px solid #d2d2d7; }
        .tts-section:last-child { border-right: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>Live Bilingual Translation & Synthesis</h1>

    <div class="controls">
        <button id="toggleBtn" class="stop-btn" onclick="toggleListen()">⏹ Stop Listening</button>
        <label for="langSelect">Target Translation:</label>
        <select id="langSelect" onchange="updateLanguage()">
            <option value="auto">Auto-Detect</option>
            <option value="fr">French</option>
            <option value="es">Spanish</option>
            <option value="de">German</option>
            <option value="it">Italian</option>
            <option value="nl">Dutch</option>
            <option value="pt">Portuguese</option>
        </select>
        <button onclick="copyToClipboard()">📋 Copy</button>
        <button onclick="saveToFile()">💾 Save As...</button>
        <button onclick="clearSession()">Reset</button>
        <button class="primary" onclick="generateAnalysis()">✨ Generate AI Summary</button>
    </div>

    <div class="tts-row">
        <div class="tts-section">
            <label>🔊 Transcript:</label>
            <select id="ttsLangTranscript">
                <option value="en">English</option>
                <option value="fr">French</option>
                <option value="es">Spanish</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
            </select>
            <button id="btnPlayTranscript" onclick="togglePlayTTS('transcript')">▶ Play</button>
            <button id="btnStopTranscript" class="stop-btn" onclick="stopTTS('transcript')" style="display:none;">⏹ Stop</button>
        </div>

        <div class="tts-section">
            <label>🔊 Summary:</label>
            <select id="ttsLangSummary">
                <option value="en">English</option>
                <option value="fr">French</option>
                <option value="es">Spanish</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
            </select>
            <button id="btnPlaySummary" onclick="togglePlayTTS('summary')">▶ Play</button>
            <button id="btnStopSummary" class="stop-btn" onclick="stopTTS('summary')" style="display:none;">⏹ Stop</button>
        </div>
    </div>

    <div class="box-title">Live Transcript (Editable):</div>
    <textarea id="transcriptBox" placeholder="Listening to MacBook microphone... Speak now. Feel free to edit text directly."></textarea>

    <div class="box-title">AI Synthesis Summary:</div>
    <div id="summaryBox" class="summary-box">Click "Generate AI Summary" to analyze the transcript.</div>
</div>

<script>
    // Audio state tracking
    const audioState = {
        transcript: { audio: null, isPlaying: false, isPaused: false },
        summary: { audio: null, isPlaying: false, isPaused: false }
    };

    function updateLanguage() {
        const lang = document.getElementById('langSelect').value;
        fetch('/api/set_language', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ language: lang })
        });
    }

    async function toggleListen() {
        await fetch('/api/toggle_listen', { method: 'POST' });
    }

    async function togglePlayTTS(target) {
        const state = audioState[target];
        const btnPlay = document.getElementById(target === 'transcript' ? 'btnPlayTranscript' : 'btnPlaySummary');
        const btnStop = document.getElementById(target === 'transcript' ? 'btnStopTranscript' : 'btnStopSummary');

        // Handle Resume
        if (state.isPaused && state.audio) {
            state.audio.play();
            state.isPaused = false;
            state.isPlaying = true;
            btnPlay.innerText = "⏸ Pause";
            return;
        }

        // Handle Pause
        if (state.isPlaying && state.audio) {
            state.audio.pause();
            state.isPaused = true;
            state.isPlaying = false;
            btnPlay.innerText = "▶ Resume";
            return;
        }

        // Handle New Playback Generation
        const langId = target === 'transcript' ? 'ttsLangTranscript' : 'ttsLangSummary';
        const lang = document.getElementById(langId).value;

        // Extract selected text if available, otherwise read all
        let text = "";
        if (target === 'transcript') {
            const transcriptBox = document.getElementById('transcriptBox');
            if (transcriptBox.selectionStart !== transcriptBox.selectionEnd) {
                text = transcriptBox.value.substring(transcriptBox.selectionStart, transcriptBox.selectionEnd);
            } else {
                text = transcriptBox.value;
            }
        } else {
            text = document.getElementById('summaryBox').innerText;
        }

        if (!text || text.trim() === "" || text.includes("Click \\"Generate")) {
            alert("No text available to read out loud.");
            return;
        }

        btnPlay.innerText = "⏳ Generating...";
        btnPlay.disabled = true;
        btnStop.style.display = "none";

        try {
            const res = await fetch('/api/tts', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text: text, language: lang, target: target })
            });
            const data = await res.json();

            if (data.audio_url) {
                btnPlay.innerText = "⏸ Pause";
                btnPlay.disabled = false;
                btnStop.style.display = "inline-block";

                state.audio = new Audio(data.audio_url);
                state.isPlaying = true;
                state.isPaused = false;

                state.audio.play();

                state.audio.onended = () => {
                    resetTTSUI(target);
                };
                state.audio.onerror = () => {
                    alert("Playback failed.");
                    resetTTSUI(target);
                };
            }
        } catch (e) {
            console.error(e);
            alert("Network error generating TTS.");
            resetTTSUI(target);
        }
    }

    function stopTTS(target) {
        const state = audioState[target];
        if (state.audio) {
            state.audio.pause();
            state.audio.currentTime = 0; // Reset to start
        }
        resetTTSUI(target);
    }

    function resetTTSUI(target) {
        const state = audioState[target];
        state.audio = null;
        state.isPlaying = false;
        state.isPaused = false;

        const btnPlay = document.getElementById(target === 'transcript' ? 'btnPlayTranscript' : 'btnPlaySummary');
        const btnStop = document.getElementById(target === 'transcript' ? 'btnStopTranscript' : 'btnStopSummary');

        btnPlay.innerText = "▶ Play";
        btnPlay.disabled = false;
        btnStop.style.display = "none";
    }

    async function pollData() {
        try {
            const res = await fetch('/api/data');
            const data = await res.json();

            const txtBox = document.getElementById('transcriptBox');

            if (data.new_segments && data.new_segments.length > 0) {
                const addedText = data.new_segments.join('\\n\\n') + '\\n\\n';
                const isScrolledToBottom = txtBox.scrollTop + txtBox.clientHeight >= txtBox.scrollHeight - 20;
                const selectionStart = txtBox.selectionStart;
                const selectionEnd = txtBox.selectionEnd;
                const isFocused = document.activeElement === txtBox;

                if (txtBox.value && !txtBox.value.endsWith('\\n') && !txtBox.value.endsWith('\\n\\n')) {
                    txtBox.value += '\\n\\n' + addedText;
                } else {
                    txtBox.value += addedText;
                }

                if (isFocused) {
                    txtBox.setSelectionRange(selectionStart, selectionEnd);
                }

                if (isScrolledToBottom) {
                    txtBox.scrollTop = txtBox.scrollHeight;
                }
            }

            if (data.summary && data.summary !== document.getElementById('summaryBox').innerText) {
                document.getElementById('summaryBox').innerText = data.summary;
            }

            const toggleBtn = document.getElementById('toggleBtn');
            if (data.is_listening) {
                toggleBtn.className = 'stop-btn';
                toggleBtn.innerText = '⏹ Stop Listening';
                if(!txtBox.value) txtBox.placeholder = "Listening to MacBook microphone... Speak now. Feel free to edit text directly.";
            } else {
                toggleBtn.className = 'start-btn';
                toggleBtn.innerText = '▶ Start Listening';
                txtBox.placeholder = "Microphone paused.";
            }

        } catch (e) {
            console.error("Polling error:", e);
        }
    }

    async function generateAnalysis() {
        document.getElementById('summaryBox').innerText = "Analyzing transcript using local Qwen2.5-14B architecture (this may take a moment to load if executed for the first time)...";
        const currentText = document.getElementById('transcriptBox').value;

        await fetch('/api/summarize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text: currentText })
        });
    }

    async function clearSession() {
        if(confirm("Clear current transcript and calculations?")) {
            await fetch('/api/clear', { method: 'POST' });
            document.getElementById('transcriptBox').value = '';
            document.getElementById('summaryBox').innerText = 'Click "Generate AI Summary" to analyze the transcript.';

            // Stop any playing audio on reset
            stopTTS('transcript');
            stopTTS('summary');
        }
    }

    function copyToClipboard() {
        const text = document.getElementById('transcriptBox').value;
        navigator.clipboard.writeText(text);
        alert("Transcript copied to clipboard buffer.");
    }

    function saveToFile() {
        const text = document.getElementById('transcriptBox').value;
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'translated_transcript.txt';
        a.click();
    }

    setInterval(pollData, 1000);
</script>
</body>
</html>
"""

def main():
    t_thread = threading.Thread(target=transcription_worker, daemon=True)
    t_thread.start()

    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 1.0)
        )
        audio_stream.start()
    except Exception as e:
        print(f"Failed to access MacBook microphonic hardware device: {e}")
        state.is_running = False
        sys.exit(1)

    server = HTTPServer(("localhost", PORT), WebServerHandler)
    print(f"\n=======================================================")
    print(f" Live Web Server actively tracking microphone details.")
    print(f" URL Link: http://localhost:{PORT}")
    print(f" Press Ctrl+C directly inside this terminal window to stop.")
    print(f"=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down local services smoothly...")
    finally:
        state.is_running = False
        audio_stream.stop()
        audio_stream.close()
        server.server_close()

if __name__ == "__main__":
    main()
