import librosa
import numpy as np
import soundfile as sf
import sofar
import os
import scipy.signal as signal
import subprocess
from pathlib import Path

def get_unique_filename(base_name):
    path = Path(base_name)
    if not path.exists(): return str(path)
    counter = 1
    while True:
        new_name = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not new_name.exists(): return str(new_name)
        counter += 1

# --- 1. User Inputs ---
print("\n--- Venue Master: 3D Multi-Device Mixer ---")
backing_path = input("Enter path to Main/Concert Track: ").strip()
lead_path    = input("Enter path to Lead Solo (Optional - Press Enter to skip): ").strip()

is_single_track = not lead_path
if is_single_track:
    print(">> Single-Track Mode Active.")
    lead_path = backing_path

# New Volume and DSP Prompts
try:
    main_vol    = float(input("Main Vol [Default 0.5]: ") or 0.5)
    spatial_vol = float(input("Spatial Vol [Default 0.9]: ") or 0.9)
    lfe_gain    = float(input("Subwoofer (LFE) Gain [Default 0.6]: ") or 0.6)
    v_width     = float(input("Arena Width [Default 2.2]: ") or 2.2)
except ValueError:
    print(">> Invalid input detected. Using default values.")
    main_vol, spatial_vol, lfe_gain, v_width = 0.5, 0.9, 0.6, 2.2

print("\n[1] Open Air Stadium (Long, airy reflections)")
print("[2] Indoor Arena (Powerful, aggressive punch)")
print("[3] Small Club (Tight, intimate echoes)")
venue_choice = input("Select Venue Preset: ")

# Venue DSP Parameters [Pre-delay, Echo Strength]
# (Width is now handled by user input above)
presets = {
    '1': [0.045, 0.40],
    '2': [0.030, 0.30],
    '3': [0.015, 0.15]
}
v_delay, v_echo = presets.get(venue_choice, presets['1'])

print("\nSelect Output Mode:")
print("[1] Generic Soundbar (Custom Channels)")
print("[2] AirPods Pro (Binaural)")
print("[3] Tesla Model X Plaid (12-Ch)")
mode = input("Choice: ")

num_channels = 8 # Default fallback
if mode == '1':
    try:
        user_ch = input("Enter number of Soundbar channels (e.g., 2, 6, 8): ").strip()
        num_channels = int(user_ch) if user_ch else 8
    except ValueError:
        print(">> Invalid input. Defaulting to 8 channels.")
        num_channels = 8

# --- 2. Load and Resample ---
target_sr = 96000 if mode == '2' else 48000
backing, sr = librosa.load(backing_path, sr=target_sr, mono=True)
lead = backing if is_single_track else librosa.load(lead_path, sr=target_sr, mono=True)[0]

max_len = max(len(backing), len(lead))
backing, lead = np.pad(backing, (0, max_len - len(backing))), np.pad(lead, (0, max_len - len(lead)))

# --- 3. Crossover & LFE ---
sos_lp = signal.butter(4, 85, 'lp', fs=sr, output='sos')
sos_hp = signal.butter(4, 85, 'hp', fs=sr, output='sos')
lfe_raw = signal.sosfilt(sos_lp, backing)
lfe_final = (lfe_raw + signal.sosfilt(signal.iirfilter(2, [35, 75], btype='bandpass', fs=sr, output='sos'), lfe_raw) * 2.5) * lfe_gain
back_hp, lead_hp = signal.sosfilt(sos_hp, backing) * main_vol, signal.sosfilt(sos_hp, lead) * main_vol

# --- 4. Rendering Mode Logic ---
if mode == '1': # DYNAMIC SOUNDBAR MODE
    output = np.zeros((max_len, num_channels))
    side = (back_hp - np.roll(back_hp, int(v_delay * sr))) * 0.5 * v_width
    echo_sig = np.roll(lead_hp, int(v_delay * sr)) * v_echo * spatial_vol

    # Mapping logic based on channel availability
    if num_channels >= 2: # Left / Right
        output[:, 0] = (back_hp + side)
        output[:, 1] = (back_hp - side)

    if num_channels >= 4: # LFE / Center Lead
        output[:, 2] = lfe_final
        output[:, 3] = (lead_hp * 0.5)

    if num_channels >= 6: # Surrounds
        output[:, 4] = side * spatial_vol
        output[:, 5] = -side * spatial_vol

    if num_channels >= 8: # Rear / Height
        output[:, 6] = (lead_hp + echo_sig) * 0.9
        output[:, 7] = (lead_hp + echo_sig) * 0.9

    if num_channels > 8: # Fill extra channels with ambient wash
        for ch in range(8, num_channels):
            output[:, ch] = echo_sig * (0.4 / (ch - 6))

    suffix = f"SOUNDBAR_{num_channels}CH"

elif mode == '3': # TESLA 12-CHANNEL
    output = np.zeros((max_len, 12))
    side_signal = (back_hp - np.roll(back_hp, int(v_delay * sr))) * 0.5 * v_width
    output[:, 0], output[:, 1] = (back_hp + side_signal), (back_hp - side_signal)
    output[:, 2], output[:, 3] = (lead_hp * 0.5), lfe_final
    output[:, 4:6], output[:, 6:8] = side_signal[:, None] * spatial_vol, side_signal[:, None] * (spatial_vol * 0.6)
    echo_sig = np.roll(lead_hp, int(v_delay * sr)) * v_echo * spatial_vol
    output[:, 8:10], output[:, 10:12] = (lead_hp + echo_sig)[:, None] * 0.7, echo_sig[:, None] * 0.5
    suffix = "TESLA_PLAID"

else: # AIRPODS BINAURAL
    sofa = sofar.read_sofa("D2_96K_24bit_512tap_FIR_SOFA.sofa")
    ir_len = sofa.Data_IR.shape[2]
    left_f, right_f = np.zeros(max_len + ir_len), np.zeros(max_len + ir_len)

    def get_hrtf(az, el):
        idx = np.argmin(np.sqrt((sofa.SourcePosition[:, 0] - az)**2 + (sofa.SourcePosition[:, 1] - el)**2))
        return sofa.Data_IR[idx, 0, :], sofa.Data_IR[idx, 1, :]

    # Apply width to the HRTF positioning
    for pos in [[45, 0], [-45, 0]]:
        il, ir = get_hrtf(pos[0], pos[1])
        left_f[:max_len+ir_len-1] += np.convolve(back_hp * 0.5, il, mode='full')
        right_f[:max_len+ir_len-1] += np.convolve(back_hp * 0.5, ir, mode='full')

    il_h, ir_h = get_hrtf(0, 60)
    lead_s = lead_hp + (np.roll(lead_hp, int(v_delay * sr)) * v_echo * spatial_vol)
    left_f[:max_len+ir_len-1] += np.convolve(lead_s, il_h, mode='full')
    right_f[:max_len+ir_len-1] += np.convolve(lead_s, ir_h, mode='full')

    output = np.vstack((left_f[:max_len] + lfe_final, right_f[:max_len] + lfe_final)).T
    suffix = "AIRPODS_3D"

# --- 5. Export and Auto-Convert ---
# Normalization logic
output = (output / (np.max(np.abs(output)) + 1e-6)) * 0.88
wav_path = get_unique_filename(f"{Path(backing_path).stem}_{suffix}.wav")
sf.write(wav_path, output, target_sr, subtype='PCM_24')

m4a_path = wav_path.replace(".wav", ".m4a")
cmd = ["ffmpeg", "-i", wav_path, "-c:a", "alac", m4a_path, "-y"]
subprocess.run(cmd)

os.remove(wav_path)
print(f"\nSUCCESS! High-Fidelity M4A created: {m4a_path}")
