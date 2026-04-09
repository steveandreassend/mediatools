import os
import subprocess

def separate_all_stems(input_file):
    print(f"--> Processing: {input_file}")
    print(f"--> Separating: Vocals, Drums, Bass, Guitar, Piano, and Other (Strings)...")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # COMMAND CHANGE:
    # Changed model from 'htdemucs_ft' to 'htdemucs_6s'.
    # This 6-stem model splits audio into:
    # 1. vocals
    # 2. drums
    # 3. bass
    # 4. guitar
    # 5. piano
    # 6. other (Strings, synths, horns, etc.)
    command = [
        "demucs",
        "-n", "htdemucs_6s",
        input_file
    ]

    try:
        # Run the command
        subprocess.run(command, check=True)

        # Calculate output paths
        filename_no_ext = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join("separated", "htdemucs_6s", filename_no_ext)

        print("\nSUCCESS! Audio separated into 6 stems.")
        print(f"Folder: {output_dir}")
        print("-" * 30)
        print(f"🎹 PIANO/KEYS:    {os.path.join(output_dir, 'piano.wav')}")
        print(f"🎸 GUITAR TRACK:  {os.path.join(output_dir, 'guitar.wav')}")
        print(f"🎻 STRINGS/OTHER: {os.path.join(output_dir, 'other.wav')}")
        print(f"🎸 BASS TRACK:    {os.path.join(output_dir, 'bass.wav')}")
        print(f"🥁 DRUMS:         {os.path.join(output_dir, 'drums.wav')}")
        print(f"🎤 VOCALS:        {os.path.join(output_dir, 'vocals.wav')}")
        print("-" * 30)

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: 'demucs' not found. Make sure it is installed with pip.")

if __name__ == "__main__":
    print("--- 6-Stem Full Band Separator ---")
    file_name = input("Enter the filename (drag & drop file here): ").strip()

    # Clean up drag-and-drop quotes
    file_name = file_name.replace("'", "").replace('"', "")

    separate_all_stems(file_name)
    