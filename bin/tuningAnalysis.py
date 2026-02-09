import librosa
import numpy as np
import sys
import os

def analyze_track_frequency(file_path):
    print(f"Analyzing: {os.path.basename(file_path)}...")
    
    # 1. Load the audio
    # distinct frequencies are easier to detect in the middle of the track,
    # so we load a duration of 60 seconds from the middle if possible.
    try:
        # Get duration first to find the middle
        duration = librosa.get_duration(filename=file_path)
        start_time = max(0, (duration / 2) - 30)
        
        y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=60)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Estimate Tuning
    # This returns the offset from A440 in fraction of a bin (semitone)
    pitch_shift = librosa.estimate_tuning(y=y, sr=sr)
    
    # 3. Calculate the Reference Frequency
    # Formula: 440 * 2^(offset / 12)
    detected_freq = 440 * (2 ** (pitch_shift / 12))
    
    # 4. Report
    print("\n" + "="*40)
    print("TUNING ANALYSIS")
    print("="*40)
    print(f"Standard Reference:  440.00 Hz")
    print(f"Detected Offset:     {pitch_shift:+.4f} semitones")
    print(f"Estimated Reference: {detected_freq:.2f} Hz")
    print("-" * 40)
    
    # 5. Interpretation
    if abs(detected_freq - 440) < 1.5:
        print("Conclusion: Standard 440Hz Tuning")
    elif abs(detected_freq - 432) < 1.5:
        print("Conclusion: 432Hz Tuning Detected")
    elif abs(detected_freq - 442) < 1.5:
        print("Conclusion: 442Hz (Orchestral/European) Tuning Detected")
    else:
        print(f"Conclusion: Non-standard tuning (~{detected_freq:.1f} Hz)")
    print("="*40 + "\n")

if __name__ == "__main__":
    # You can hardcode a path here or use input()
    target_file = input("Enter path to audio file: ").strip().strip('"')
    if os.path.exists(target_file):
        analyze_track_frequency(target_file)
    else:
        print("File not found.")