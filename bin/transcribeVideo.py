import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from xml.etree.ElementTree import ParseError
from pydub import AudioSegment
import speech_recognition as sr
import yt_dlp
import heapq

# Hardcoded stopwords
stop_words = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
])

def simple_sent_tokenize(text):
    # Simple regex for sentence tokenization
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

def simple_word_tokenize(text):
    # Simple word tokenization using regex
    return re.findall(r'\b\w+\b', text.lower())

def summarize(text, ratio=0.2):
    sentences = simple_sent_tokenize(text)
    if len(sentences) < 5:
        return text
    word_freq = {}
    for sentence in sentences:
        for word in simple_word_tokenize(sentence):
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
    if not word_freq:
        return text
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in simple_word_tokenize(sentence):
            if word in word_freq:
                sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]
    num_sentences = max(1, int(len(sentences) * ratio))
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sentences[i] for i in sorted(summary_sentences)])
    return summary

def get_transcript(video_url, language='en'):
    video_id = video_url.split("v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Find preferred transcript
        try:
            transcript = transcript_list.find_transcript([language])
        except NoTranscriptFound:
            print(f"No transcript in {language}. Available transcripts:")
            for t in transcript_list:
                print(f" - {t.language_code} ({t.language}) {'(generated)' if t.is_generated else ''}")
            language = input("Enter the language code you want to use: ")
            transcript = transcript_list.find_transcript([language])
        # Fetch the transcript
        try:
            transcript_data = transcript.fetch()
            return video_id, "\n".join([entry['text'] for entry in transcript_data])
        except ParseError:
            print("Parse error while fetching transcript. The transcript may be empty or invalid.")
            return video_id, None
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return video_id, None
    except NoTranscriptFound:
        print("No transcripts found for this video.")
        return video_id, None
    except Exception as e:
        print(f"Unexpected error fetching transcript: {e}")
        return video_id, None

def transcribe_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    recognizer = sr.Recognizer()
    transcript = ""
    chunk_length_ms = 60000  # 60 seconds
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk.export("temp_chunk.wav", format="wav")
        with sr.AudioFile("temp_chunk.wav") as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            transcript += text + " "
        except sr.UnknownValueError:
            print(f"Could not understand audio chunk starting at {i/1000} seconds")
        except sr.RequestError as e:
            print(f"Could not request results for chunk starting at {i/1000} seconds: {e}")
    if os.path.exists("temp_chunk.wav"):
        os.remove("temp_chunk.wav")
    return transcript.strip()

def main():
    video_url = input("Enter the YouTube video URL: ")
    video_id, transcript = get_transcript(video_url)
    if transcript is None:
        print("Failed to fetch YouTube subtitles. Attempting audio transcription...")
        # Download audio
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
            # Optimize audio
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(44100).set_channels(1)
            audio.export("optimized_audio.wav", format="wav")
            transcript = transcribe_audio("optimized_audio.wav")
            print("Transcript from audio file:")
            print(transcript)
            with open("audio_transcript.txt", "w") as file:
                file.write(transcript)
            print("Transcript saved to audio_transcript.txt")
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists("optimized_audio.wav"):
                os.remove("optimized_audio.wav")
        except Exception as e:
            print(f"Failed to download or transcribe audio: {e}")
            return
    else:
        print("Transcript from YouTube subtitles:")
        print(transcript)
        with open(f"{video_id}.txt", "w") as file:
            file.write(transcript)
        print(f"Transcript saved to {video_id}.txt")

    # Generate and print summary
    if transcript:
        summary = summarize(transcript)
        print("\nSummary:")
        print(summary)
        with open(f"{video_id}_summary.txt", "w") as file:
            file.write(summary)
        print(f"Summary saved to {video_id}_summary.txt")

if __name__ == "__main__":
    main()
