import subprocess
import sys
from pathlib import Path

def convert_videos_to_m4a():
    # Prompt for the directory path
    dir_input = input("Enter the directory path containing MP4 or MOV files: ").strip()
    target_dir = Path(dir_input).expanduser().resolve()

    # Validate directory
    if not target_dir.is_dir():
        print(f"Error: The directory '{target_dir}' does not exist.")
        sys.exit(1)

    # Find all MP4 and MOV files (case-insensitive search)
    video_files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.mov']]

    if not video_files:
        print("No MP4 or MOV files found in the specified directory.")
        return

    print(f"\nFound {len(video_files)} video file(s). Starting conversion...\n")

    for video_file in video_files:
        # Set up the output file path (.m4a extension)
        m4a_file = video_file.with_suffix('.m4a')

        # Parse the filename (ARTIST - SONG)
        # .stem gets the filename without the extension
        filename_clean = video_file.stem
        parts = filename_clean.split(" - ", 1)

        artist = ""
        title = ""

        if len(parts) == 2:
            artist = parts[0].strip()
            title = parts[1].strip()
        else:
            print(f"Warning: '{filename_clean}' does not match 'ARTIST - SONG' format. Skipping metadata.")
            title = filename_clean # Fallback to using the whole filename as the title

        # Construct the FFmpeg command
        # -vn: disables video extraction
        # -c:a copy: copies the audio stream without re-encoding
        # -y: overwrites output files without asking
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_file),
            "-vn",
            "-c:a", "copy"
        ]

        # Add metadata flags if parsing was successful
        if artist:
            cmd.extend(["-metadata", f"artist={artist}"])
        if title:
            cmd.extend(["-metadata", f"title={title}"])

        cmd.append(str(m4a_file))

        print(f"Processing: {video_file.name}")

        try:
            # Run ffmpeg, suppressing standard output/error for a cleaner console
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  -> Saved as {m4a_file.name} (Artist: '{artist}', Title: '{title}')")

        except subprocess.CalledProcessError:
            print(f"  -> Error processing {video_file.name}. Ensure the file contains a valid audio track.")
        except FileNotFoundError:
            print("\nError: 'ffmpeg' is not installed or not found in your system's PATH.")
            print("Install it via Homebrew: brew install ffmpeg")
            sys.exit(1)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    convert_videos_to_m4a()
