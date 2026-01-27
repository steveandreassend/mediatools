import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. User Input ---
input_file = input("Enter path to your Multichannel WAV: ").strip()
if not os.path.exists(input_file):
    exit("File not found.")

# --- 2. Load Multichannel Data ---
print("Analyzing 3D Soundstage and Channel Levels...")
audio, sr = librosa.load(input_file, sr=None, mono=False)

if audio.ndim < 2:
    exit("This visualizer requires a multichannel file (5.1 or 7.1).")

# --- 3. Terminal Level Metering ---
labels = ["1: FL (Front Left)", "2: FR (Front Right)", "3: LFE (Subwoofer)",
          "4: FC (Center)", "5: SL (Surround L)", "6: SR (Surround R)",
          "7: HL (Height L)", "8: HR (Height R)"]

print("\n" + "="*45)
print(f"{'CHANNEL':<20} | {'PEAK (dB)':<10} | {'RMS (dB)':<10}")
print("-" * 45)

channel_energy = []
for i in range(len(labels)):
    if i < len(audio):
        ch_data = audio[i]

        # Calculate Peak and RMS in Decibels
        peak = np.max(np.abs(ch_data))
        rms = np.sqrt(np.mean(ch_data**2))

        # Convert to dB (avoid log of zero)
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        rms_db = 20 * np.log10(rms) if rms > 0 else -100

        channel_energy.append(rms)
        print(f"{labels[i]:<20} | {peak_db:>8.2f} dB | {rms_db:>8.2f} dB")
    else:
        channel_energy.append(0)
        print(f"{labels[i]:<20} | {'EMPTY':>11} | {'EMPTY':>11}")

print("="*45 + "\n")

# --- 4. 3D Plotting Preparation ---
speaker_coords = [
    [30, 0], [ -30, 0], [0, -90], [0, 0], [110, 0], [-110, 0], [45, 45], [-45, 45]
]

# Normalize energy for the visual orbs
norm_energy = np.array(channel_energy)
if np.max(norm_energy) > 0:
    norm_energy = (norm_energy / np.max(norm_energy)) * 100

def polar_to_cartesian(az, el, radius):
    az_rad, el_rad = np.radians(az), np.radians(el)
    x = radius * np.cos(el_rad) * np.sin(az_rad)
    y = radius * np.cos(el_rad) * np.cos(az_rad)
    z = radius * np.sin(el_rad)
    return x, y, z

# --- 5. Generate 3D Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Background Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs, ys, zs = 50*np.cos(u)*np.sin(v), 50*np.cos(u)*np.cos(v), 50*np.sin(v)
ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.1)

# Plot Speakers
short_labels = ["FL", "FR", "SUB", "C", "SL", "SR", "HL", "HR"]
for i, (az, el) in enumerate(speaker_coords):
    x, y, z = polar_to_cartesian(az, el, 50)
    size = norm_energy[i] * 12
    color = 'blue' if el > 0 else ('red' if el < 0 else 'green')

    if size > 1: # Only plot active channels
        ax.scatter(x, y, z, s=size, c=color, alpha=0.7, edgecolors='w')
        ax.text(x, y, z, short_labels[i], color='black', fontsize=10, fontweight='bold')

# Listener
ax.scatter(0, 0, 0, color='black', s=100, marker='^')
ax.set_title(f"3D Energy Map: {os.path.basename(input_file)}")
ax.view_init(elev=20, azim=45)

plt.show()
