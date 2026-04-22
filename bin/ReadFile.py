import os
import subprocess

# Suppress common HuggingFace/PyTorch terminal noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from styletts2 import tts
except ImportError:
    print("Error: styletts2 is not installed. Please run 'python3.10 -m pip install styletts2'.")
    exit(1)

def prompt_for_file_mac():
    """Opens a native macOS file picker using AppleScript."""
    applescript = '''
    try
        set theFile to choose file with prompt "Select a text file to speak:" of type {"public.plain-text"}
        POSIX path of theFile
    on error
        return ""
    end try
    '''

    # Run the AppleScript command
    result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)

    # Clean and return the path
    return result.stdout.strip()

def main():
    print("Opening Mac Finder prompt...")

    # 1. Prompt the user
    filepath = prompt_for_file_mac()

    # Handle the case where the user clicks "Cancel"
    if not filepath:
        print("No file selected or prompt cancelled. Exiting.")
        return

    print(f"Selected file: {filepath}")

    # 2. Read the text
    with open(filepath, 'r', encoding='utf-8') as file:
        text_to_speak = file.read().strip()

    if not text_to_speak:
        print("Error: The provided text file is empty.")
        return

    print("\nLoading StyleTTS 2 model...")

    # 3. Initialize the TTS engine
    my_tts = tts.StyleTTS2()

    output_wav = "output_speech.wav"
    print(f"\nSynthesizing audio... Saving to {output_wav}")

    # Generate the audio
    my_tts.inference(text_to_speak, output_wav_file=output_wav)

    print(f"Success! The audio has been saved to: {os.path.abspath(output_wav)}")

    # 4. Play the audio natively using macOS's built-in afplay command
    print("Playing audio...")
    os.system(f"afplay '{output_wav}'")

if __name__ == "__main__":
    main()
