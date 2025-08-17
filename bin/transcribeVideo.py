import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from xml.etree.ElementTree import ParseError
from pydub import AudioSegment
import speech_recognition as sr
import yt_dlp

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
    chunk_length_ms = 30000  # 30 seconds to reduce timeout risks
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
    transcript_filename = f"{video_id}_transcript.txt"
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
            with open(transcript_filename, "w") as file:
                file.write(transcript)
            print(f"Transcript saved to {transcript_filename}")
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
        with open(transcript_filename, "w") as file:
            file.write(transcript)
        print(f"Transcript saved to {transcript_filename}")

if __name__ == "__main__":
    main()
