import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# --- 1. User Input ---
input_file = input("Enter path to your Multichannel WAV: ").strip()
if not os.path.exists(input_file):
    exit("File not found.")

# --- 2. Load Audio ---
print("Loading for Real-Time Analysis...")
audio, sr = librosa.load(input_file, sr=None, mono=False)
if audio.ndim < 2:
    exit("Requires a multichannel file.")

# Animation parameters
fps = 10
frame_duration = 1.0 / fps
samples_per_frame = int(sr * frame_duration)
total_frames = audio.shape[1] // samples_per_frame

# Speaker Layout (Sony Fixed Mapping)
speaker_coords = [
    [30, 0], [-30, 0], [0, -90], [0, 0], [110, 0], [-110, 0], [45, 45], [-45, 45]
]
labels = ["FL", "FR", "SUB", "C", "SL", "SR", "HL", "HR"]

# --- 3. Setup Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def polar_to_cartesian(az, el, radius):
    az_rad, el_rad = np.radians(az), np.radians(el)
    x = radius * np.cos(el_rad) * np.sin(az_rad)
    y = radius * np.cos(el_rad) * np.cos(az_rad)
    z = radius * np.sin(el_rad)
    return x, y, z

# Initialize orbs
scatters = []
for i, (az, el) in enumerate(speaker_coords):
    x, y, z = polar_to_cartesian(az, el, 50)
    color = 'blue' if el > 0 else ('red' if el < 0 else 'green')
    sc = ax.scatter([x], [y], [z], s=10, c=color, alpha=0.7, edgecolors='w')
    ax.text(x, y, z, labels[i], fontsize=10, fontweight='bold')
    scatters.append(sc)

ax.scatter(0, 0, 0, color='black', s=100, marker='^') # Listener
ax.view_init(elev=20, azim=45)
ax.set_axis_off()

# --- 4. Animation Update ---
def update(n):
    start = n * samples_per_frame
    end = start + samples_per_frame
    chunk = audio[:, start:end]
    
    # Calculate RMS for each channel in this chunk
    rms_values = []
    output_str = f"\rFrame {n}/{total_frames} | "
    
    for i in range(len(labels)):
        if i < chunk.shape[0]:
            rms = np.sqrt(np.mean(chunk[i]**2))
            rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100
            
            # Update Orb Size (Scale: 0 to 1500)
            size = np.clip((rms_db + 60) * 25, 10, 2000)
            scatters[i]._sizes = [size]
            
            # Print meter logic (brief)
            if i in [2, 6, 7]: # Focus on Sub and Heights
                output_str += f"{labels[i]}: {rms_db:>5.1f}dB "
        
    print(output_str, end="", flush=True)
    return scatters

# Run Animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)
plt.title(f"LIVE 3D Map: {os.path.basename(input_file)}")
plt.show()