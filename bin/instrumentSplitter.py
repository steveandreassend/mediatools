import os
import subprocess
import numpy as np
from scipy.io import wavfile

def separate_lead_and_rhythm(guitar_file, output_dir):
    """
    Splits a stereo guitar track into Center (Lead) and Wide (Rhythm),
    outputting both as properly phased stereo files.
    """
    print("--> Processing Mid/Side split for Lead/Rhythm...")
    
    if not os.path.exists(guitar_file):
        print("Error: guitar.wav not found.")
        return None, None, None
        
    try:
        sample_rate, data = wavfile.read(guitar_file)
        original_dtype = data.dtype
        
        # Convert to float32 for clean math without overflow
        data = data.astype(np.float32)
        
        if len(data.shape) == 2 and data.shape[1] == 2:
            left = data[:, 0]
            right = data[:, 1]
            
            # Mid Channel (Center/Lead)
            mid = (left + right) / 2.0
            # Duplicate mid to L and R for standard stereo playback
            lead_stereo = np.column_stack((mid, mid))
            
            # Side Channel (Wide/Rhythm)
            side = (left - right) / 2.0
            # Phase invert the right side to maintain the wide stereo image
            rhythm_stereo = np.column_stack((side, -side))
            
            lead_path = os.path.join(output_dir, 'guitar_lead_center.wav')
            rhythm_path = os.path.join(output_dir, 'guitar_rhythm_sides.wav')
            
            # Helper function to normalize audio and prevent clipping
            def write_normalized(path, audio_array):
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    if np.issubdtype(original_dtype, np.integer):
                        info = np.iinfo(original_dtype)
                        scaled = (audio_array / max_val) * (info.max * 0.95)
                        wavfile.write(path, sample_rate, scaled.astype(original_dtype))
                    else:
                        scaled = (audio_array / max_val) * 0.95
                        wavfile.write(path, sample_rate, scaled.astype(np.float32))
                else:
                    wavfile.write(path, sample_rate, audio_array.astype(original_dtype))

            write_normalized(lead_path, lead_stereo)
            write_normalized(rhythm_path, rhythm_stereo)
            
            print(f"🎸 LEAD GUITAR (Center):   {lead_path}")
            print(f"🎸 RHYTHM GUITAR (Sides):  {rhythm_path}")
            
            # Return these so the backing track mixer can use them
            return sample_rate, rhythm_stereo, original_dtype
        else:
            print("Guitar track is mono. Cannot perform Mid/Side split.")
            return None, None, None
            
    except Exception as e:
        print(f"Failed to process audio for lead/rhythm split: {e}")
        return None, None, None

def create_backing_track(output_dir, sample_rate, rhythm_stereo, original_dtype):
    """
    Mixes all instrumental and vocal stems with the rhythm guitar 
    to create a single minus-lead backing track.
    """
    print("--> Mixing 'Minus-Lead' Backing Track...")
    stems_to_mix = ['drums.wav', 'bass.wav', 'piano.wav', 'other.wav', 'vocals.wav']
    
    # Start the mix with the isolated rhythm guitar track
    mixed_audio = np.copy(rhythm_stereo)
    
    for stem in stems_to_mix:
        stem_path = os.path.join(output_dir, stem)
        if os.path.exists(stem_path):
            sr, data = wavfile.read(stem_path)
            data = data.astype(np.float32)
            
            # Ensure array lengths match exactly before summing
            min_len = min(len(mixed_audio), len(data))
            mixed_audio[:min_len] += data[:min_len]
            
    output_path = os.path.join(output_dir, 'Backing_Track_Minus_Lead.wav')
    
    # Normalize the final mix to 95% to prevent master bus clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            scaled = (mixed_audio / max_val) * (info.max * 0.95)
            wavfile.write(output_path, sample_rate, scaled.astype(original_dtype))
        else:
            scaled = (mixed_audio / max_val) * 0.95
            wavfile.write(output_path, sample_rate, scaled.astype(np.float32))
    else:
        wavfile.write(output_path, sample_rate, mixed_audio.astype(original_dtype))
        
    print(f"🎧 FULL BACKING TRACK:     {output_path}")

def separate_all_stems(input_file):
    print(f"--> Processing: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    command = [
        "demucs",
        "-n", "htdemucs_6s",
        "--shifts", "2", 
        "--overlap", "0.25",
        input_file
    ]

    try:
        subprocess.run(command, check=True)

        filename_no_ext = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join("separated", "htdemucs_6s", filename_no_ext)

        print("\nSUCCESS! Audio separated into 6 stems.")
        print("-" * 30)
        
        # Execute the split function and capture the rhythm array
        guitar_stem = os.path.join(output_dir, 'guitar.wav')
        sample_rate, rhythm_stereo, dtype = separate_lead_and_rhythm(guitar_stem, output_dir)
        
        # Generate the backing track if the split was successful
        if rhythm_stereo is not None:
            create_backing_track(output_dir, sample_rate, rhythm_stereo, dtype)
            
        print("-" * 30)
        print(f"All files saved to: {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing Demucs: {e}")
    except FileNotFoundError:
        print("Error: 'demucs' not found. Make sure it is installed with pip.")

if __name__ == "__main__":
    print("--- Minus-Lead Backing Track Generator ---")
    file_name = input("Enter the filename (drag & drop file here): ").strip()
    file_name = file_name.replace("'", "").replace('"', "")
    separate_all_stems(file_name)