import os
import subprocess

def separate_all_stems(input_file):
    print(f"--> Processing: {input_file}")
    print(f"--> Separating: Vocals, Drums, Bass, and Guitar...")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # COMMAND CHANGE:
    # We removed "--two-stems=vocals".
    # By default, Demucs splits audio into 4 parts:
    # 1. Vocals
    # 2. Drums
    # 3. Bass
    # 4. Guitar (This is your Guitar/Piano/Synths)
    command = [
        "demucs",
        "-n", "htdemucs_ft",
        input_file
    ]

    try:
        # Run the command
        subprocess.run(command, check=True)

        # Calculate output paths
        filename_no_ext = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join("separated", "htdemucs_ft", filename_no_ext)

        print("\nSUCCESS! Audio separated into 4 stems.")
        print(f"Folder: {output_dir}")
        print("-" * 30)
        print(f"🎸 GUITAR TRACK:  {os.path.join(output_dir, 'guitar.wav')}")
        print(f"🎹 BASS TRACK:    {os.path.join(output_dir, 'bass.wav')}")
        print(f"🥁 DRUMS (Trash): {os.path.join(output_dir, 'drums.wav')}")
        print(f"🎤 VOCALS (Trash):{os.path.join(output_dir, 'vocals.wav')}")
        print("-" * 30)
        print("Tip: Drag 'guitar.wav' into GarageBand. If it sounds thin, try layering 'bass.wav' underneath it.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: 'demucs' not found. Make sure you installed it with pip.")

if __name__ == "__main__":
    print("--- Full Band Separator (Vocals/Drums/Bass/Guitar) ---")
    file_name = input("Enter the filename (drag & drop file here): ").strip()

    # Clean up drag-and-drop quotes
    file_name = file_name.replace("'", "").replace('"', "")

    separate_all_stems(file_name)
