import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter # NEW IMPORTS REQUIRED

# --- Audio Setup ---
SAMPLE_RATE = 44100
BPM = 120
BEAT_DURATION = 60.0 / BPM  # 0.5 seconds per beat

# --- Frequencies (in Hertz) ---
NOTES = {
    'F#2': 92.50,
    'B2': 123.47,
    'C#3': 138.59,
    'D3': 146.83,
    'E3': 164.81,
    'E#3': 174.61, # Enharmonic F3
    'F#3': 185.00,
    'G3': 196.00,
    'G#3': 207.65,
    'A3': 220.00,
    'B3': 246.94,
    'C#4': 277.18,
    'D4': 293.66,
    'E4': 329.63
}

# --- Chord Definitions ---
c_Fsm = ['F#3', 'A3', 'C#4']
c_D = ['D3', 'F#3', 'A3']
c_Bm = ['B2', 'D3', 'F#3']
c_E = ['E3', 'G#3', 'B3']
c_Cs_Es = ['E#3', 'G#3', 'C#4'] # C#/E#
c_E_Gs = ['G#3', 'B3', 'E4']    # E/G#
c_A = ['A3', 'C#4', 'E4']
c_Css4 = ['C#3', 'F#3', 'G#3']  # C#sus4
c_Cs = ['C#3', 'E#3', 'G#3']    # C#
c_Fs5 = ['F#2', 'C#3', 'F#3']   # F#5 Power chord
c_G = ['G3', 'B3', 'D4']
c_Em = ['E3', 'G3', 'B3']
c_Csm = ['C#3', 'E3', 'G#3']
c_Rest = []                     # Rest (Silence)

# --- Song Sections (Notes, Duration in Beats) ---
intro_loop = [(c_Fsm, 4), (c_D, 4), (c_Bm, 4), (c_E, 2), (c_Cs_Es, 2)]
intro_end = [(c_Fsm, 2), (c_E_Gs, 2), (c_A, 2), (c_D, 2), (c_Css4, 4), (c_Cs, 4)]

chorus_loop = [(c_Fsm, 4), (c_D, 4), (c_Bm, 4), (c_E, 4)]
chorus_end = [(c_Fsm, 2), (c_E_Gs, 2), (c_A, 2), (c_D, 2), (c_Css4, 4), (c_Cs, 4)]

fs5_4_measures = [(c_Fs5, 16)]
fs5_2_measures = [(c_Fs5, 8)]

verse_body = [
    (c_Fsm, 12), (c_Bm, 4), (c_Fsm, 8),
    (c_Fsm, 2), (c_E_Gs, 2), (c_A, 4), (c_D, 4), (c_E, 4),
    (c_A, 2), (c_E_Gs, 2), (c_Fsm, 2), (c_E, 2), (c_D, 4), (c_Csm, 4)
]

# The complex syncopated rhythm block for the "E" hits spanning 2 measures (8 beats)
syncopated_E_hits = [
    (c_Rest, 0.5), (c_E, 0.5),                # Beat 1
    (c_E, 0.5), (c_E, 0.5),                   # Beat 2
    (c_Rest, 0.5), (c_E, 1.5),                # Beats 3 & 4 (Eighth tied across barline)
    (c_E, 1.0), (c_Rest, 1.0), (c_Rest, 2.0)  # Next Measure: Tied completion, Quarter rest, Half rest
]

guitar_solo = [
    (c_Bm, 4), (c_A, 4), (c_D, 4), (c_G, 4), (c_Em, 4), (c_A, 4), (c_Bm, 4), (c_Fsm, 4),
    (c_Bm, 4), (c_A, 4), (c_D, 4), (c_G, 4), (c_Em, 4), (c_A, 4), (c_Bm, 4), (c_Cs, 4)
]

# --- Build the Full Sequence ---
sequence = []
sequence.extend(intro_loop * 4)
sequence.extend(intro_end)

sequence.extend(chorus_loop * 2)
sequence.extend(chorus_end)
sequence.extend(fs5_4_measures)

# Verse 1
sequence.extend(verse_body)
sequence.extend(syncopated_E_hits)

# Chorus
sequence.extend(chorus_loop * 2)

# Verse 2
sequence.extend(verse_body)
sequence.extend(syncopated_E_hits)

# Chorus & Intro to Solo
sequence.extend(chorus_loop * 2)
sequence.extend(chorus_end)
sequence.extend(fs5_2_measures)

# Solo & Instrumental
sequence.extend(guitar_solo)
sequence.extend(chorus_loop * 2) # Instrumental is same as chorus loop
sequence.extend(chorus_end)

# Final Chorus
sequence.extend(chorus_loop * 2)
sequence.extend(chorus_end)
sequence.extend([(c_Fsm, 4)]) # Final Chord

def butter_lowpass_filter(data, cutoff, sample_rate, order=4):
    """Creates and applies a digital low-pass filter to the audio data."""
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def generate_warm_pad(freq, duration, sample_rate=SAMPLE_RATE):
    """Generates a filtered, dual-oscillator sawtooth pad."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Oscillator 1: Base Sawtooth
    osc1 = 2.0 * (t * freq - np.floor(t * freq + 0.5))

    # Oscillator 2: Slightly detuned Sawtooth (adds thickness/chorus effect)
    detune_freq = freq * 1.008  # Detuned by a fraction of a percent
    osc2 = 2.0 * (t * detune_freq - np.floor(t * detune_freq + 0.5))

    # Mix oscillators
    mixed_wave = (osc1 + osc2) * 0.5

    # Apply Low-Pass Filter to remove harsh buzz and create warmth
    # A cutoff around 800 Hz to 1200 Hz works well for warm pads.
    cutoff_frequency = 900.0
    filtered_wave = butter_lowpass_filter(mixed_wave, cutoff_frequency, sample_rate, order=2)

    return filtered_wave

def apply_pad_envelope(audio_data, sample_rate=SAMPLE_RATE):
    """Applies a slow Attack and long Release for smooth, swelling chords."""
    attack_time = 0.35   # Slow swell in
    release_time = 0.50  # Long fade out

    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)

    total_samples = len(audio_data)

    # Handle short notes where the envelope is longer than the note itself
    if total_samples < attack_samples + release_samples:
        half_len = total_samples // 2
        attack_samples = half_len
        release_samples = half_len

    envelope = np.ones_like(audio_data)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    return audio_data * envelope

def make_chord(notes, duration):
    """Calculates waves for individual notes and sums them into a warm pad chord."""
    chord_wave = np.zeros(int(SAMPLE_RATE * duration))

    if not notes:
        return chord_wave

    for note in notes:
        freq = NOTES[note]
        # Use the new warm pad generator
        note_wave = generate_warm_pad(freq, duration)
        chord_wave += note_wave

    chord_wave = chord_wave / len(notes)

    # Use the new slow envelope
    chord_wave = apply_pad_envelope(chord_wave)
    return chord_wave

# --- Track Generation ---
track = np.array([])

print("Synthesizing full song progression. This may take a moment...")
for chord_notes, beats in sequence:
    duration = BEAT_DURATION * beats
    wave = make_chord(chord_notes, duration)
    track = np.concatenate((track, wave))

# --- Export to WAV ---
max_amplitude = 32767
master_volume = 0.6

track_int16 = np.int16(track * max_amplitude * master_volume)

output_filename = "synth_chords_full_song.wav"
wavfile.write(output_filename, SAMPLE_RATE, track_int16)
print(f"✅ Audio successfully synthesized and saved to '{output_filename}'")
