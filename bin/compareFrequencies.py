import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def profile_comparison():
    # 1. Prompt for file names
    file_440 = input("Enter the path/name of the ORIGINAL (440Hz) file: ").strip()
    file_432 = input("Enter the path/name of the CONVERTED (432Hz) file: ").strip()

    # Check if files exist
    if not os.path.exists(file_440) or not os.path.exists(file_432):
        print("\nError: One or both files could not be found. Please check the filenames and try again.")
        return

    print("\nLoading files and analyzing frequencies... (this may take a moment)")

    # 2. Load both files (sr=None preserves the original sample rate)
    try:
        y440, sr440 = librosa.load(file_440, sr=None)
        y432, sr432 = librosa.load(file_432, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 3. Compute Spectrum for both
    # High n_fft (8192) provides the resolution needed to see the 8Hz difference
    n_fft = 8192

    spec_440 = np.abs(librosa.stft(y440, n_fft=n_fft))
    spec_432 = np.abs(librosa.stft(y432, n_fft=n_fft))

    # Convert to average power (dB)
    mean_440 = librosa.amplitude_to_db(np.mean(spec_440, axis=1), ref=np.max)
    mean_432 = librosa.amplitude_to_db(np.mean(spec_432, axis=1), ref=np.max)

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr440, n_fft=n_fft)

    # 4. Plotting
    plt.figure(figsize=(14, 7))

    # Overlay both spectra
    plt.plot(freqs, mean_440, label=f'Original (440Hz): {file_440}', alpha=0.7, color='#1f77b4')
    plt.plot(freqs, mean_432, label=f'Shifted (432Hz): {file_432}', alpha=0.8, color='#ff7f0e', linestyle='--')

    # Zoom into the "Guitar Range" (typically 80Hz to 1200Hz for solos)
    plt.xlim(100, 1200)
    plt.ylim(-60, 0)

    plt.title('Comparison of 440Hz vs 432Hz Frequency Peaks', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Relative Power (dB)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    print("Analysis complete. Displaying graph...")
    plt.show()

    # 5. Programmatic Tuning Verification
    tuning_440 = librosa.estimate_tuning(y=y440, sr=sr440)
    tuning_432 = librosa.estimate_tuning(y=y432, sr=sr432)

    print(f"\n" + "="*30)
    print(f"TUNING ESTIMATION RESULTS")
    print(f"="*30)
    print(f"Original File Offset: {tuning_440:+.2f} cents")
    print(f"Shifted File Offset:  {tuning_432:+.2f} cents")
    print(f"Measured Difference:  {abs(tuning_440 - tuning_432)*100:.2f} cents")
    print(f"Target Difference:    ~31.77 cents")
    print("="*30)

if __name__ == "__main__":
    profile_comparison()
