import os
import shutil
import subprocess
import sys
import tempfile
import time

import librosa
import numpy as np
import soundfile as sf

# ==========================================
# GLOBAL CONFIGURATION & PATHS
# ==========================================
REPO_DIR = "/Users/stephenandreassend/git/mediatools/bin/zfturbo_repo"
MODEL_DIR = "/Users/stephenandreassend/git/mediatools/bin/models"

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"

def resolve_file_path(input_val, default_dir):
    if os.path.exists(input_val):
        return os.path.abspath(input_val)
    filename = input_val if input_val.endswith('.wav') else f"{input_val}.wav"
    return os.path.join(default_dir, filename)


# ==========================================
# 1. GENERIC ROFORMER EXTRACTION
# ==========================================
def extract_specialized_stems_roformer(raw_input_wav, final_output_dir, model_file, config_file, target_stems, high_quality=False):
    inference_script = os.path.join(REPO_DIR, "inference.py")
    model_path = os.path.join(MODEL_DIR, model_file)
    config_path = os.path.join(MODEL_DIR, config_file)

    stems_display = ", ".join([s.capitalize() for s in target_stems])
    print(f"    🌟 Extracting {stems_display} via BS-Roformer...")

    input_basename = os.path.splitext(os.path.basename(raw_input_wav))[0]
    roformer_subfolder = os.path.join(final_output_dir, input_basename)

    with tempfile.TemporaryDirectory() as temp_in_dir:
        temp_input_wav = os.path.join(temp_in_dir, os.path.basename(raw_input_wav))
        shutil.copy2(raw_input_wav, temp_input_wav)

        cmd = [
            sys.executable,
            inference_script,
            "--model_type", "bs_roformer",
            "--start_check_point", model_path,
            "--config_path", config_path,
            "--input_folder", temp_in_dir,
            "--store_dir", final_output_dir,
            "--force_cpu"
        ]

        if high_quality:
            cmd.extend(["--bigshifts", "1", "--use_tta"])

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            for stem_name in target_stems:
                source_wav = os.path.join(roformer_subfolder, f"{stem_name}.wav")
                final_wav = os.path.join(final_output_dir, f"{stem_name}.wav")

                if os.path.exists(source_wav):
                    audio, sr = sf.read(source_wav)
                    max_amp = np.max(np.abs(audio))
                    if max_amp > 0:
                        normalized_audio = audio / max_amp
                        sf.write(final_wav, normalized_audio, sr, subtype='PCM_16')
                    else:
                        shutil.move(source_wav, final_wav)
                    print(f"    ✅ Successfully extracted and normalized {stem_name}.")
                else:
                    print(f"    ⚠️ Expected output {source_wav} not found.")

            if os.path.exists(roformer_subfolder):
                shutil.rmtree(roformer_subfolder)
                print(f"    🧹 Cleaned up remaining unrequested Roformer stems.")

            return True

        except subprocess.CalledProcessError as e:
            print(f"    ❌ Roformer Inference failed for {model_file}: {e}")
            return False


# ==========================================
# 2. MDX23C ORCHESTRAL EXTRACTION
# ==========================================
def extract_mdx23c_ensemble(raw_input_wav, final_output_dir, model_file, config_file, high_quality=False):
    inference_script = os.path.join(REPO_DIR, "inference.py")
    model_path = os.path.join(MODEL_DIR, model_file)
    config_path = os.path.join(MODEL_DIR, config_file)

    print(f"    🎻 Extracting Orchestral Ensemble via MDX23C...")
    if high_quality:
        print("    ℹ️  Note: High Quality Mode (TTA) is bypassed for MDX23C to prevent known phase-cancellation bugs. Running safe standard inference.")

    input_basename = os.path.splitext(os.path.basename(raw_input_wav))[0]
    mdx_subfolder = os.path.join(final_output_dir, input_basename)

    with tempfile.TemporaryDirectory() as temp_in_dir:
        temp_input_wav = os.path.join(temp_in_dir, os.path.basename(raw_input_wav))
        shutil.copy2(raw_input_wav, temp_input_wav)

        cmd = [
            sys.executable,
            inference_script,
            "--model_type", "mdx23c",
            "--start_check_point", model_path,
            "--config_path", config_path,
            "--input_folder", temp_in_dir,
            "--store_dir", final_output_dir,
            "--force_cpu"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            if os.path.exists(mdx_subfolder):
                generated_files = [f for f in os.listdir(mdx_subfolder) if f.endswith('.wav')]
                target_file_name = next((f for f in generated_files if "orch" in f.lower()), None)

                if target_file_name:
                    source_wav = os.path.join(mdx_subfolder, target_file_name)
                    final_wav = os.path.join(final_output_dir, "orchestra_ensemble.wav")

                    audio, sr = sf.read(source_wav)
                    max_amp = np.max(np.abs(audio))
                    if max_amp > 0:
                        normalized_audio = audio / max_amp
                        sf.write(final_wav, normalized_audio, sr, subtype='PCM_16')
                    else:
                        shutil.move(source_wav, final_wav)
                    print(f"    ✅ Successfully extracted and normalized orchestra_ensemble.wav")
                else:
                    print(f"    ⚠️ Expected 'orch.wav' not found in MDX23C output.")

                shutil.rmtree(mdx_subfolder)
                print(f"    🧹 Cleaned up remaining MDX temp files.")
            else:
                 print(f"    ⚠️ ZFTurbo output directory was not created.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ❌ MDX23C Inference failed: {e}")
            return False


# ==========================================
# 3. DEMUCS MAIN SEPARATION
# ==========================================
def separate_all_stems(input_file, base_output_dir):
    print(f"    🥁 Running standard 6-stem Demucs on {os.path.basename(input_file)}...")
    print(f"       (Extracting Vocals, Bass, Drums, and composite tracks)")
    try:
        command = ["demucs", "-n", "htdemucs_6s", "-o", base_output_dir, input_file]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Demucs failed: {e}")
        return False


# ==========================================
# 4. GUITAR PROCESSING & PURIFICATION
# ==========================================
def spectral_bleed_subtraction(target_path, subtract_path, output_path, strength=0.3):
    try:
        y_target, sr = librosa.load(target_path, sr=None, mono=False)
        y_sub, _ = librosa.load(subtract_path, sr=sr, mono=False)

        if y_target.ndim == 1: y_target = np.vstack((y_target, y_target))
        if y_sub.ndim == 1: y_sub = np.vstack((y_sub, y_sub))

        min_len = min(y_target.shape[1], y_sub.shape[1])
        y_target = y_target[:, :min_len]
        y_sub = y_sub[:, :min_len]

        clean_target = np.zeros_like(y_target)

        for ch in range(y_target.shape[0]):
            stft_target = librosa.stft(y_target[ch])
            stft_sub = librosa.stft(y_sub[ch])
            mag_target, phase_target = librosa.magphase(stft_target)
            mag_sub, _ = librosa.magphase(stft_sub)
            mag_clean = np.maximum(mag_target - (mag_sub * strength), 0.0)
            clean_target[ch] = librosa.istft(mag_clean * phase_target, length=min_len)

        sf.write(output_path, clean_target.T, sr)
        return True
    except Exception as e:
        print(f"    ❌ Spectral Subtraction failed: {e}")
        return False

def isolate_pure_guitar(input_file, final_output_file):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = ["demucs", "-n", "htdemucs_6s", "--two-stems", "guitar", "-o", temp_dir, input_file]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            demucs_guitar = os.path.join(temp_dir, "htdemucs_6s", base_name, "guitar.wav")

            if os.path.exists(demucs_guitar):
                shutil.copy2(demucs_guitar, final_output_file)
                return True
            return False
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Demucs pure guitar isolation failed: {e}")
        return False

def preprocess_guitar_input(input_path, output_path, hq_mode):
    print(f"    🧹 Pre-processing {os.path.basename(input_path)} with BS-Roformer to filter out other instruments...")

    temp_dir = os.path.join(os.path.dirname(output_path), f"temp_rofo_guitar_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    abs_input = os.path.abspath(input_path)

    model_file = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
    config_file = "mvsep_mega_model_bs_roformer_53_stems_v1.yaml"

    success = extract_specialized_stems_roformer(
        raw_input_wav=abs_input,
        final_output_dir=temp_dir,
        model_file=model_file,
        config_file=config_file,
        target_stems=["electric-guitar", "acoustic-guitar"],
        high_quality=hq_mode
    )

    if success:
        elec_path = os.path.join(temp_dir, "electric-guitar.wav")
        acou_path = os.path.join(temp_dir, "acoustic-guitar.wav")
        combined, sr_val = None, None

        def load_audio(p):
            a, s = librosa.load(p, sr=None, mono=False)
            if a.ndim == 1: a = np.vstack((a, a))
            return a, s

        if os.path.exists(elec_path):
            combined, sr_val = load_audio(elec_path)

        if os.path.exists(acou_path):
            acou_audio, acou_sr = load_audio(acou_path)
            if combined is not None:
                min_len = min(combined.shape[1], acou_audio.shape[1])
                combined = combined[:, :min_len] + acou_audio[:, :min_len]
            else:
                combined = acou_audio
                sr_val = acou_sr

        if combined is not None:
            max_amp = np.max(np.abs(combined))
            if max_amp > 0:
                combined = combined / max_amp
            sf.write(output_path, combined.T, sr_val, subtype='PCM_16')
            shutil.rmtree(temp_dir)
            print(f"    ✅ Guitar pre-processed successfully via 53-stem BS-Roformer.")
            return True

    print(f"    ⚠️ BS-Roformer guitar pre-processing failed. Falling back to Demucs pure guitar isolation.")
    fallback_success = isolate_pure_guitar(abs_input, output_path)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    return fallback_success

def center_side_split(input_path, center_path, side_path):
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        if audio.ndim == 1 or audio.shape[0] == 1:
            print("    ⚠️ Input track is single-channel mono. Center/Side split requires a stereo file.")
            return False

        left, right = audio[0], audio[1]
        center, side = (left + right) / 2.0, (left - right) / 2.0

        if np.max(np.abs(side)) < 1e-6:
            print("    ⚠️ Input track is dual-mono (Left and Right channels are identical). Cannot perform center/side split.")
            return False

        sf.write(side_path, np.vstack((side, side)).T, sr)
        temp_center_path = input_path.replace(".wav", "_temp_mid.wav").replace(".mp3", "_temp_mid.wav")
        sf.write(temp_center_path, np.vstack((center, center)).T, sr)

        print("    🧹 Spectrally purifying Center Lead track...")
        spectral_bleed_subtraction(temp_center_path, side_path, center_path, strength=0.8)

        if os.path.exists(temp_center_path):
            os.remove(temp_center_path)

        return True
    except Exception as e:
        print(f"    ❌ Center/Side split failed: {e}")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n🎸 HYBRID STEM SEPARATOR (Setup Phase) 🎸")
    print("-------------------------------------------------")

    file_name = input("Enter the path to your raw audio file: ").strip().replace("'", "").replace('"', "")

    if not os.path.exists(file_name):
        print("❌ File not found. Exiting.")
        sys.exit(1)

    abs_input_file = os.path.abspath(file_name)
    run_demucs = input("Run Demucs base 6-stem separation? (y/n) [y]: ").strip().lower() != 'n'

    print("\n--- High-Fidelity Specialized Extraction ---")
    print("Select the instruments you want to extract with pristine quality.")
    print("You can select multiple by separating with commas (e.g., 2,3,9).")
    print("  1 = Keyboard / Piano (6-stem ViperX)")
    print("  2 = Acoustic Guitar  (53-stem Mega Model)")
    print("  3 = Saxophone        (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print("  4 = Brass            (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print("  5 = Cello            (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print("  6 = Viola            (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print("  7 = Violin           (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print("  8 = Synth            (53-stem Mega Model)")
    print("  9 = Strings          (Triple-Pass: 53-stem -> MDX23C -> 53-stem)")
    print(" 10 = Drums            (6-stem ViperX)")
    print(" 11 = Bass Guitar      (6-stem ViperX)")
    print("  0 = Skip this phase")
    choices = input("Enter your choices [0]: ").strip() or "0"

    use_hq_mode = False
    if choices != "0":
        use_hq_mode = input("Enable Highest Quality Mode (TTA & BigShifts - much slower)? (y/n) [n]: ").strip().lower() == 'y'

    print("\n--- Optional: Post-Extraction Bleed Cleanup ---")
    run_cleanup = input("Run spectral bleed cleanup? (y/n) [n]: ").strip().lower() == 'y'

    cleanup_target, cleanup_source, cleanup_strength = "", "", 0.3
    if run_cleanup:
        cleanup_target = input("  Stem to clean (e.g., 'orchestra_ensemble' or full path): ").strip().replace("'", "").replace('"', "")
        cleanup_source = input("  Stem to subtract from it (e.g., 'guitar' or full path): ").strip().replace("'", "").replace('"', "")
        try:
            cleanup_strength = float(input("  Strength of subtraction [default 0.3]: ") or 0.3)
        except ValueError:
            pass

    print("\n--- Electric Guitar Processing ---")
    print("  1 = Skip")
    print("  2 = Tandem Subtraction (Requires separate Lead & Rhythm tracks)")
    print("  3 = Center/Side Split (Extracts center-panned lead from stereo rhythm)")
    guitar_mode = input("Select guitar processing method [1]: ").strip() or "1"

    tandem_lead, tandem_rhythm = "", ""
    bleed_r2l, bleed_l2r = 0.3, 0.7
    cs_target_track = ""

    abs_tandem_lead, abs_tandem_rhythm = "", ""

    if guitar_mode == "2":
        tandem_lead = input("Enter path to the LEAD-heavy track: ").strip().replace("'", "").replace('"', "")
        tandem_rhythm = input("Enter path to the RHYTHM-heavy track: ").strip().replace("'", "").replace('"', "")
        if os.path.exists(tandem_lead): abs_tandem_lead = os.path.abspath(tandem_lead)
        if os.path.exists(tandem_rhythm): abs_tandem_rhythm = os.path.abspath(tandem_rhythm)
        try:
            bleed_r2l = float(input("Strength to remove Rhythm from Lead track [default 0.3]: ") or 0.3)
            bleed_l2r = float(input("Strength to remove Lead from Rhythm track [default 0.7]: ") or 0.7)
        except ValueError: pass
    elif guitar_mode == "3":
        cs_target_track = input("Enter path to the stereo guitar track (leave blank to use Demucs 'guitar.wav'): ").strip().replace("'", "").replace('"', "")

    print("\n🚀 Setup Complete. Starting Unattended Processing...\n")

    # --- EXECUTION PHASE ---
    base_out = "separated"
    filename_no_ext = os.path.splitext(os.path.basename(file_name))[0]
    out_dir = os.path.join(base_out, "htdemucs_6s", filename_no_ext)
    os.makedirs(out_dir, exist_ok=True)
    abs_out_dir = os.path.abspath(out_dir)

    step_times = {}
    total_start_time = time.time()

    # 1. Demucs Base Separation
    if run_demucs:
        print("--- Phase 1: Base Separation ---")
        t0 = time.time()
        separate_all_stems(file_name, base_out)
        step_times["Demucs 6-Stem Separation"] = time.time() - t0
    else:
        print("--- Phase 1: Base Separation (SKIPPED) ---")

    # 2. Advanced Extractions (Roformer & MDX23C Triple-Pass)
    if choices != "0":
        mega_ckpt = "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
        mega_yaml = "mvsep_mega_model_bs_roformer_53_stems_v1.yaml"
        mdx_ckpt = "model_mdx23c_ep_54_sdr_4.0870.ckpt"
        mdx_yaml = "config_orchestra_mdx23c.yaml"

        model_configs = {
            "1": {"type": "direct", "model": "BS-Rofo-SW-Fixed.ckpt", "yaml": "BS-Rofo-SW-Fixed.yaml", "stem": "piano"},
            "2": {"type": "direct", "model": mega_ckpt, "yaml": mega_yaml, "stem": "acoustic-guitar"},
            "3": {"type": "triple", "stem": "saxophone"},
            "4": {"type": "triple", "stem": "brass"},
            "5": {"type": "triple", "stem": "cello"},
            "6": {"type": "triple", "stem": "viola"},
            "7": {"type": "triple", "stem": "violin"},
            "8": {"type": "direct", "model": mega_ckpt, "yaml": mega_yaml, "stem": "synth"},
            "9": {"type": "triple", "stem": "strings"},
            "10": {"type": "direct", "model": "BS-Rofo-SW-Fixed.ckpt", "yaml": "BS-Rofo-SW-Fixed.yaml", "stem": "drums"},
            "11": {"type": "direct", "model": "BS-Rofo-SW-Fixed.ckpt", "yaml": "BS-Rofo-SW-Fixed.yaml", "stem": "bass"},
        }

        selected_options = [c.strip() for c in choices.split(",")]
        direct_tasks = {}
        run_triple = False
        triple_stems = []

        for option in selected_options:
            if option in model_configs:
                conf = model_configs[option]
                if conf["type"] == "direct":
                    task_key = (conf["model"], conf["yaml"])
                    if task_key not in direct_tasks: direct_tasks[task_key] = []
                    direct_tasks[task_key].append(conf["stem"])
                elif conf["type"] == "triple":
                    run_triple = True
                    triple_stems.append(conf["stem"])

        if direct_tasks:
            print("\n--- Phase 1.5A: Direct Roformer Extraction ---")
            t0 = time.time()
            for (mod_file, yaml_file), stems in direct_tasks.items():
                extract_specialized_stems_roformer(file_name, out_dir, mod_file, yaml_file, stems, use_hq_mode)
            step_times["Direct Roformer Extraction"] = time.time() - t0

        if run_triple:
            print("\n--- Phase 1.5B: Orchestral Triple-Pass Extraction ---")
            pass1_dir = os.path.join(out_dir, "triple_pass_1")
            os.makedirs(pass1_dir, exist_ok=True)

            print("    🎯 PASS 1: Initial Split from Raw Mix (53-Stem Roformer)...")
            t0 = time.time()
            extract_specialized_stems_roformer(file_name, pass1_dir, mega_ckpt, mega_yaml, triple_stems, use_hq_mode)
            step_times["Triple-Pass (Pass 1: Split)"] = time.time() - t0

            print("    🎯 PASS 2 & 3: MDX23C Purification and Final Roformer Clean...")
            t0 = time.time()
            for stem in triple_stems:
                pass1_file = os.path.join(pass1_dir, f"{stem}.wav")
                if os.path.exists(pass1_file):
                    print(f"\n    🎷 Processing '{stem}'...")
                    pass2_dir = os.path.join(out_dir, f"triple_pass_2_{stem}")
                    os.makedirs(pass2_dir, exist_ok=True)

                    print(f"       -> PASS 2: Applying MDX23C to increase SDR...")
                    if extract_mdx23c_ensemble(pass1_file, pass2_dir, mdx_ckpt, mdx_yaml, use_hq_mode):
                        pass2_file = os.path.join(pass2_dir, "orchestra_ensemble.wav")
                        if os.path.exists(pass2_file):
                            print(f"       -> PASS 3: Final clean split from high-SDR stem...")
                            extract_specialized_stems_roformer(pass2_file, out_dir, mega_ckpt, mega_yaml, [stem], use_hq_mode)

                    if os.path.exists(pass2_dir):
                        shutil.rmtree(pass2_dir)
                else:
                    print(f"    ⚠️ Expected Pass 1 output {pass1_file} missing. Skipping.")

            if os.path.exists(pass1_dir):
                shutil.rmtree(pass1_dir)
            step_times["Triple-Pass (Pass 2 & 3)"] = time.time() - t0

    # 2.5 Optional Bleed Cleanup
    if run_cleanup and cleanup_target and cleanup_source:
        print("\n--- Phase 1.6: Stem Bleed Cleanup ---")
        t0 = time.time()
        target_file = resolve_file_path(cleanup_target, out_dir)
        source_file = resolve_file_path(cleanup_source, out_dir)

        if os.path.exists(target_file) and os.path.exists(source_file):
            print(f"    🧹 Cleaning '{os.path.basename(target_file)}' by subtracting '{os.path.basename(source_file)}'...")
            temp_clean = os.path.join(out_dir, "temp_cleaned_output.wav")
            if spectral_bleed_subtraction(target_file, source_file, temp_clean, cleanup_strength):
                shutil.move(temp_clean, target_file)
                print(f"    ✅ Successfully cleaned {os.path.basename(target_file)}.")
        else:
            if not os.path.exists(target_file): print(f"    ⚠️ Missing Target: {target_file}")
            if not os.path.exists(source_file): print(f"    ⚠️ Missing Source: {source_file}")
        step_times["Stem Bleed Cleanup"] = time.time() - t0

    # 3. Guitar Processing
    if guitar_mode == "2":
        print("\n--- Phase 2: Electric Guitar Processing (Tandem Subtraction) ---")
        t0 = time.time()

        pure_lead_input = os.path.join(out_dir, "temp_pure_lead_input.wav")
        pure_rhythm_input = os.path.join(out_dir, "temp_pure_rhythm_input.wav")

        preprocess_guitar_input(tandem_lead, pure_lead_input, use_hq_mode)
        preprocess_guitar_input(tandem_rhythm, pure_rhythm_input, use_hq_mode)

        lead_pure_path = os.path.join(out_dir, "guitar_lead_pure.wav")
        rhythm_pure_path = os.path.join(out_dir, "guitar_rhythm_pure.wav")

        print("    🎸 Step 1: Processing Lead Tandem Subtraction on Purified Stems...")
        spectral_bleed_subtraction(pure_lead_input, pure_rhythm_input, lead_pure_path, strength=bleed_r2l)

        print("    🎸 Step 2: Processing Rhythm Tandem Subtraction on Purified Stems...")
        spectral_bleed_subtraction(pure_rhythm_input, pure_lead_input, rhythm_pure_path, strength=bleed_l2r)

        if os.path.exists(pure_lead_input): os.remove(pure_lead_input)
        if os.path.exists(pure_rhythm_input): os.remove(pure_rhythm_input)
        print("    ✅ Lead and Rhythm Guitars successfully isolated and purified.")
        step_times["Guitar Tandem Processing"] = time.time() - t0

    elif guitar_mode == "3":
        print("\n--- Phase 2: Electric Guitar Processing (Center/Side Split) ---")
        t0 = time.time()
        input_target = cs_target_track if cs_target_track else os.path.join(out_dir, "guitar.wav")

        pure_cs_input = os.path.join(out_dir, "temp_pure_cs_input.wav")
        preprocess_guitar_input(input_target, pure_cs_input, use_hq_mode)

        center_out = os.path.join(out_dir, "guitar_center_lead.wav")
        side_out = os.path.join(out_dir, "guitar_side_rhythm.wav")

        if os.path.exists(pure_cs_input):
            print(f"    🎸 Splitting purified {os.path.basename(input_target)} into Center and Side...")
            if center_side_split(pure_cs_input, center_out, side_out):
                print("    ✅ Center (Lead) and Side (Rhythm) guitars successfully split.")
            os.remove(pure_cs_input)
        else:
            print(f"    ❌ Purified target track not found.")
        step_times["Guitar Center/Side Split"] = time.time() - t0

    total_elapsed_time = time.time() - total_start_time

    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY REPORT")
    print("="*60)

    print("\n📥 INPUT FILES:")
    print(f"   - Main Mix: {abs_input_file}")
    if guitar_mode == "2":
        print(f"   - Tandem Lead: {abs_tandem_lead or 'File not found'}")
        print(f"   - Tandem Rhythm: {abs_tandem_rhythm or 'File not found'}")

    print("\n⏱️  PROCESSING TIMES:")
    for step_name, duration in step_times.items():
        print(f"   - {step_name}: {format_time(duration)}")
    print(f"   > TOTAL ELAPSED TIME: {format_time(total_elapsed_time)}")

    print("\n📁 FINAL OUTPUT FILES:")
    if os.path.exists(abs_out_dir):
        generated_files = [f for f in os.listdir(abs_out_dir) if f.endswith('.wav')]
        if generated_files:
            for f in sorted(generated_files):
                print(f"   - {os.path.join(abs_out_dir, f)}")
        else:
            print("   - No .wav files found in the output directory.")
    else:
        print("   - Output directory was not created.")

    print("="*60 + "\n")
