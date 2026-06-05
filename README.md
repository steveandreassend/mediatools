# Media Tools Repo

This repository contains Python and shell scripts to run on your laptop that:
* Enhance your holiday photos and videos. Upscale to 4K or 8K resolution. Correct blurry images. Fix closed eyes.
* Create coloring-in pictures from your photo memories that you can print out.
* Download videos from websites and save in high quality video and audio.
* Transcribe and summarize content (videos, audio, websites, docs) that is online or offline to make executive summaries using your chosen LLM running in Ollama.
* Shrink the size of your iCloud camera roll by archiving your photos and videos to USB. Free up space on your phone and iCloud.
* Examine the frequency of recordings when converting from 440Hz to 432Hz.
* Examine the guitar tuning of an audio file.
* Generate 3D sound to replicate Dolby Atmos effects by combining audio tracks.
* Split a music track into separate wav files for each instrument.
  
## Scripts

### coloringPicture.py
Converts any holiday photo into a printable coloring-in picture for kids, allowing them to relive memories creatively.

- **Pro Tip 1**: Print the coloring-in picture with the original photo in the corner to guide kids on color choices.
- **Pro Tip 2**: For imaginary themes, use Grok (not ChatGPT or Gemini) to generate a photo of the desired scene, then process it with this script for printing.

### coloringPicturePDF.py
As above, and giving the option to create a PDF with a thumbnail of the original image on the top-left corner to serve as a coloring-in guide. Works with photos in either landscape or portait mode to make best use of the PDF page real estate.

### imageUpscaler.py
Upscales holiday photos to 4K or 8K resolution using a deep learning neural network to fill gaps and enhance faces, delivering sharp, high-quality results even on older iPhone devices.

- **Features**: Increases resolution, enhances faces, and applies additional quality-improving techniques, even for some blurry photos.
- **Benefit**: Zoom in without losing clarity, potentially eliminating the need for a new high-end camera phone.
- **Requirements**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use. CPU mode is available but significantly slower.

### videoUpscaler.py
Upscales holiday videos frame by frame to improve quality, with batch processing for efficiency.

- **Requirements**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use. CPU mode is available but significantly slower.
- **Warning**: Running on a basic laptop in CPU mode may result in long processing times.

### eyeOpener.py
Fixes closed eyes in photos by using a trained model. Prompts the user to select the face in the image, and describe the characteristics and lighting effects to render the eyes.

- **Requirements**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use. CPU mode is available but significantly slower.

### videoDownloader.sh
Downloads any online video in the highest quality as a portable MP4 file for offline viewing on any device.

- **Use Case**: No ads, cookies, mobile data, or tracking, ensuring a clean and private experience.

### transcribeSummarize.py
This script provides an automated, locally processed pipeline designed to extract and summarize the core content of any YouTube video without relying on paid external APIs. It begins by accepting a YouTube URL via the command line or a prompt and attempts to fetch the native YouTube subtitles; if none are available, it automatically downloads the video's audio, optimizes the track, and transcribes it locally using the Apple Silicon-optimized mlx_whisper. Once the text is secured, the script leverages a local Ollama server running Llama 3 to condense the transcript—intelligently chunking and recursively processing videos longer than 5,000 words—into a well-formatted Executive Summary and a detailed list of Key Points. Finally, it saves both the full raw transcript and the generated summary into clean, readable text files while automatically deleting any temporary audio files to keep your working directory tidy. Verified with python3.10.

- **Use Case**: Save time by summarizing long videos with an executive summary with a list of the key takeaways - PRIVATELY on your local Mac, ACCURATELY without consulting external sources, and for FREE without requiring a subscription fee. It covers the situations where ChatGPT (et al.) either i) refuses to make a summary of a video when there are no transcripts on the video, or ii) if it does, it peppers the summary with information from external sources that may distort the output.

- **Pro Tip 1**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use and at least 24GB of system memory.
- **Pro Tip 2**: The Ollama server must be running with the Llama3 model. Start it with 'ollama serve'.
- **Pro Tip 3**: Adapt the code if you prefer to use a remote cloud-based LLM to perform the document summarization.
- **Pro Tip 4**: You can substantiate Llama3 with the model of your choice.

### TranscribeSummarizeRead.py
This script provides an end-to-end, locally processed pipeline designed to extract, summarize, and audibly play back the core concepts of any YouTube video without relying on paid external APIs. It begins by accepting a YouTube URL and attempting to fetch native subtitles; if none are available, it automatically downloads the video's audio, optimizes it, and transcribes it locally using Apple Silicon-optimized mlx_whisper. Once the transcript is generated, the script leverages a local Ollama server running Llama 3 to condense the text—intelligently chunking and processing videos longer than 5,000 words—into a formatted Executive Summary and a list of Key Points. This summary is then synthesized into high-fidelity, natural-sounding audio using the StyleTTS 2 engine, complete with built-in stability patches for seamless execution. Finally, the generated audio is launched in a custom PyQt6 floating media player that allows for interactive playback, including speed adjustments and skipping, before the script automatically cleans up any temporary files to keep your working directory tidy. Verified with python3.10.

Updated to use a standard media player progress bar and defaults to a faster 1.2x playback speed.

- **Use Case**: Save time by summarizing long videos with an executive summary with a list of the key takeaways - without having to read it. Just listen with it running in the background.

### ReadFile.py
Simple TTS tool. Opens a file selector prompt on MacOS to read aloud any text file. Uses StyleTTS 2 to generate a realistic human voice.

### summarizeDocument.py
Generates an executive summary of a document using Ollama with Meta's Llama3 model for summarization. Works with Word docx, PDF, or plain text files. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing documents with an executive summary with a list of the key takeaways, again with the same benefits as transcribeSummarize.py.
- **Pro Tip**: Use the script benchmarkSummarization.sh to run a stress test on MacOS of the summarizeDocument.py performance.

### summarizePage.py
Generates an executive summary of a webpage using Ollama with Meta's Llama3 model for summarization. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing webpages with an executive summary with a list of the key takeaways, again with the same benefits as transcribeSummarize.py.

### summarizeAudio.py
Generates an executive summary of an audio file using Ollama with Meta's Llama3 model for summarization. Requires python 3.10 or later.
Updated to use Whisper for performing a local transcription instead of the Google API. Better accuracy and performance.

- **Use Case**: Save time by summarizing an audio file of a long talk or seminar with an executive summary with a list of the key takeaways, again with the same benefits as transcribeSummarize.py.

### PDF2Text.py
Extracts text from a PDF file to plain text and stores it in a file. If it detects that text has been saved as images in the PDF, it uses OCR to scan the images for the text. It prompts if you want an executive summary of the text using Ollama, again with the same benefits as transcribeSummarize.py.

- **Use Case**: Solves the problem where you need to copy text in a PDF but it has been saved as images. This is often the case with PDFs containing scanned text. And it does it with the same benefits as transcribeSummarize.py.

### iCloudDeduplicate.py

- **Use Case**: When bulk exporting iCloud Photo Library media (photos, videos, metadata) to local storage or USB, this script will rename each file using a unique SHA256 hash. It also renames any associated XMP or AAE files with the media file. When merging backups with older backups, it ensures only one unique version of a media file exists.

- **Pro Tip 1**: Specify the number of parallel threads to use to make it run faster. Do not exceed the number of CPU cores on your Mac.
- **Pro Tip 2**: When backing up to archive your iCloud content offline, be sure to duplicate the backup storage on separate physical devices to guard against device failure. Store these devices in separate secure physical locations. Do not use these devices for any other purpose. Backup to one device only, and clone it to the other device(s). A Mac is best to use for running iCloud backups, not your iOS devices.
- **Pro Tip 3**: For external SSD storage, use a device with at least USB 3.2 Gen 2 speeds to achieve the best performance for large file transfers.
- **Pro Tip 4**: Check the logs for any errors and resolve them.
- **Pro Tip 5**: Run the iCloud export periodically (quarterly, bi-annually, annually) to keep your Photo Library size in check. Only purge online media from your iCloud Library once files are backed up and duplicated across at least two physical devices.
- **Pro Tip 6**: Apple devices run faster when there are fewer photos and videos stored in iCloud. This is most pronounced on older slower Apple devices. This is caused by the sheer number of files stored in iCloud and the background intelligence processing that Apple runs locally on your device of each file.
- **Pro Tip 7**: Be sure to export the original files using filename and the XMP option. The XMP and AAE files included with each media file will allow you to import the metadata into iCloud to restore the last state of the media including any edits.

### iCloudGroupBy.py

- **Use Case**: When exporting iCloud Photo Library media (photos, videos, metadata) to local storage or USB, this script will group photos in sub-folders according to the preferred date grouping. Options are yearly, quarterly, monthly, weekly, daily. This avoids having one giant folder that is slow to access and cumbersome to navigate.

- **Pro Tip 1**: Quarterly or monthly grouping will provide about the right amount of granularity for it to remain practical.
- **Pro Tip 2**: Specify the number of parallel threads to use to make it run faster. Do not exceed the number of CPU cores on your Mac.
- **Pro Tip 3**: Check the logs for any errors and resolve them.
- **Pro Tip 4**: When performing periodic iCloud backups or archiving operations, this script will allow you to merge new backups into the master archive.

### compareFrequencies.py
Prompts for two media files to compare the signal strength and frequencies.

- **Use Case**: Use after converting music to 432Hz "Verdi Tuning" to analyze the results. Displays a chart comparing the two files.

### stadium_rock.py
Converts standard audio tracks into immersive 3D soundscapes tailored for specific venues and hardware, ranging from massive stadium echoes to intimate club vibes.

- **Requirements**:
* librosa: Used for loading, resampling, and analyzing audio tracks.
* numpy: Handles the complex array mathematics and signal padding.
* soundfile: Required for writing the high-fidelity 24-bit PCM WAV files.
* sofar: Essential for reading the .sofa files used in the AirPods Pro (Binaural) 3D rendering mode.
* scipy: Specifically uses scipy.signal for applying the Butterworth filters and Crossover logic.
* Download D2_96K_24bit_512tap_FIR_SOFA.sofa from https://www.york.ac.uk/sadie-project/database.html for Air Pods Pro output mode.

- **Use Case**: It can process single media files to add a 3D effect, or it can combine a backing track and a lead guitar track (for example) to create a 3D concert effect.
- **Features**: Includes presets for Open Air Stadiums, Indoor Arenas, and Small Clubs, with adjustable parameters for arena width and subwoofer (LFE) gain.
- **Hardware Optimization**: Offers specialized output modes for generic Soundbars (custom channels), 12-channel Tesla Model S/X Plaid systems (22 speaker system), and HRTF-based binaural rendering for Apple AirPods Pro.
- **Output**: Automatically processes audio through crossover filters to enhance the low-end and exports the final mix as a high-fidelity M4A (ALAC) file.
- **Pro Tip**: If you lack a separate "Lead Solo" track, simply press Enter at the prompt to activate "Single-Track Mode," which adapts your main track for the spatial processing.

### model3Daudio.py
Analyzes and visualizes the static 3D soundstage and energy distribution of multichannel audio files, such as 5.1 or 7.1 surround mixes.

- **Use Case**: For examinining the 3D effect of applying stadium_rock.py to apply a 3D concert effect to an audio track.
- **Features**: Provides detailed terminal-based metering for Peak and RMS decibel levels across all channels, including Height and Surround speakers.
- **Visuals**: Generates a 3D energy map using a spherical wireframe where colored orbs represent active channels; the size of each orb corresponds to its signal strength.
- **Requirements**: This visualizer requires a multichannel WAV file to accurately map the spatial energy.

### monitor3Daudio.py
Provides a live, animated 3D visualization of multichannel audio to monitor energy levels and spatial movement in real-time.

- **Visuals**: Features an animated plot where speaker orbs pulse and resize dynamically based on the instantaneous RMS values of the audio frame.
- **Pro Tip 1**: Best for observing how audio energy shifts between speakers during playback, with real-time terminal monitoring specifically for Subwoofer and Height channels.
- **Pro Tip 2**: The animation uses a fixed Sony speaker mapping, making it a great tool for verifying the "height" and "punch" of your 3D mixes.

### tuningAnalysis.py
Analyzes an audio track to determine the tuning.

- **Use Case**: Determine whether a backing track is in standard 440Hz guitar tuning so that you can adapt your instrument accordingly.

### instrumentSplitter.py
Splits a music track into separate wav files for each instrument. This has been revised to enhance the splitting of guitars into rhythm and lead. And it generates a single backing track for lead guitar that can be used independently, or combined with other tracks to create a richer sound. Updated to optionally use an additional model to extract the brass and wind orchestral instruments to separate stems.

- **Use Case**: Useful for creating backing tracks, decomposing music, or remixing.

### translate_office.py
Translates any Word (DOCX, DOC), Powerpoint (PPTX), PDF, or Excel (XLSX) document from one language (autodetected) to another. Preserves all formatting and formulas. Runs locally on your laptop - private, secure, fast.

- **Use Case**: Excel and Powerpoint only allow you translate selected text one at a time. Only Word will translate the whole document.

### convert_mp4_m4a.py
Extracts the audio from all MP4 or MOV file in a specified directory and saves as M4A in the same audio quality. Assumes the file name format is ARTIST - SONG.* and sets the file metadata on the audio file.

### imageTouchup.py
Replicates the functionality of the Apple Photos image touch-up feature in newer Macs and iPhones.

- **Use Case**: Select an image and correct artifacts and defects.

### imageTouchup.py
Replicates the functionality of the Apple Photos image touch-up feature in newer Macs and iPhones.

### The_Final_Countdown_keyboard.py
Generates a wav audio file of the keyboard track for The Final Countdown arranged for The Grand Jam using the sheet music. Recreates a classic 1980s softened sawtooth synthesizer tone.

- **How?**: NumPy acts as the mathematical synthesizer engine, generating the actual soundwaves from scratch. By mapping the sheet music notes to specific mathematical frequencies, NumPy uses high-speed vector operations to calculate millions of data points representing raw oscillators (like a classic sawtooth wave). It then applies mathematical arrays acting as volume envelopes and filters to shape the harsh waves into a softened 1980s synth tone. Finally, the script uses scipy.io to package these massive arrays of calculated amplitudes into a polished, playable .wav audio file.

### instrumentSplitModel.py
Hybrid AI Stem Separator: Architecture & Processing Pipeline

This script is an advanced, multi-stage audio separation pipeline designed for local execution on macOS. It bridges the gap between generalized base separation and specialized, cloud-tier extraction by orchestrating three distinct neural network architectures. Rather than relying on a single model, the script routes audio through dynamic, multi-pass pipelines based on the specific acoustic properties of the target instruments.

#### 🧠 AI Models Employed

The pipeline leverages three state-of-the-art AI architectures, utilizing the `zfturbo` inference engine:

1. **Demucs (htdemucs_6s):**
   * **Role:** The foundational workhorse. 
   * **Function:** Performs the initial, rapid 6-stem base separation (Vocals, Bass, Drums, Guitar, Piano, Other). It provides the core stems and acts as a robust fallback for isolation tasks.
2. **BS-Roformer (Band-Split Roformer):**
   * **Role:** Surgical, high-fidelity extraction.
   * **Variants Used:**
     * *ViperX 6-Stem:* Used for pristine Drum, Bass, and Piano extraction.
     * *MVSEP 53-Stem Mega Model:* Used for complex harmonics like Acoustic Guitar, Electric Guitar, and Synths.
   * **Function:** Operates on the raw mix (or pre-filtered stems) to extract specific instruments with exceptionally high Signal-to-Distortion Ratios (SDR).
3. **MDX23C (MDX-Net):**
   * **Role:** Harmonic grouping and ensemble extraction.
   * **Variant Used:** Orchestral Ensemble Model.
   * **Function:** Highly specialized at identifying bowed and blown harmonics. Because it groups all orchestral instruments together (capturing unwanted bleed from vocals or classical percussion), it is strictly utilized as a "pre-filter" within the multi-pass pipeline rather than a final output generator.

#### ⚙️ Core Processing Approaches

The script is divided into automated phases that handle varying levels of acoustic complexity.

#### Phase 1: Base Separation
A standard Demucs pass that splits the entire mix into six foundational stems. This provides immediate, highly usable tracks and sets up fallback stems for later phases.

#### Phase 1.5: High-Fidelity Extraction
Users can selectively target specific instruments for pristine extraction. The script automatically routes the request through one of two methodologies:

* **Direct Roformer Extraction:** For rhythm section instruments (Drums, Bass, Piano, Synth), the raw audio is fed directly into the BS-Roformer models. 
* **The Triple-Pass Orchestral Architecture:** Classical and jazz instruments (Strings, Cello, Violin, Brass, Saxophone) suffer heavily from vocal and guitar bleed in standard extractions. To solve this, the script executes a studio-grade triple-pass filter:
  1. **Pass 1 (Split):** The 53-stem Roformer attempts an initial, broad extraction of the target instrument from the raw mix.
  2. **Pass 2 (Purify):** The output of Pass 1 is fed into the MDX23C Orchestral model. This acts as a strict harmonic gate, stripping away modern instrument bleed (like guitars mistaken for strings) and drastically increasing the SDR.
  3. **Pass 3 (Isolate):** The highly purified, high-SDR ensemble output from Pass 2 is fed *back* into the 53-stem Roformer for a final, surgically clean extraction of the specific instrument.

#### Phase 1.6: Spectral Bleed Cleanup
A phase-aware mathematical purification tool. Using STFT (Short-Time Fourier Transform) via `librosa`, this algorithm allows for manual spectral subtraction. If drum transients bleed into a strings track, the script calculates the exact frequencies of the drum stem and punches them out of the strings stem, leaving the target audio intact.

#### Phase 2: Advanced Electric Guitar Processing
Electric guitars present unique challenges due to distortion harmonics and stereo widening. The script features a dedicated guitar processing engine designed to isolate rhythm and lead parts, ready for direct import into DAWs like GarageBand.

* **Pre-Processing (Purification):** Before any mathematical subtraction occurs, target guitar tracks are forced through the 53-stem BS-Roformer. This guarantees that metronome clicks, drum bleed, and vocal overlaps are stripped away, ensuring all subsequent math operates only on pure guitar frequencies.
* **Tandem Subtraction:** For users with separate "Lead-heavy" and "Rhythm-heavy" practice mixes. The script spectrally subtracts the overlapping frequencies between the two files to leave behind perfectly isolated pure Lead and pure Rhythm stems.
* **Center/Side Splitting:** For standard stereo mixes. It calculates the Side channel (L - R) to extract wide-panned rhythm guitars. To overcome the mathematical limitation of Mid/Side processing (where the Center channel retains mono rhythm bleed), the script applies heavy spectral subtraction, mathematically punching the newly isolated Rhythm track out of the Center track to reveal the pure Lead solo.

#### 🛡️ Built-In Failsafes

* **TTA/Phase Cancellation Protection:** Test-Time Augmentation (Highest Quality Mode) is automatically permitted for Roformer models but hard-blocked for MDX23C inference to prevent known mathematical phase-cancellation bugs.
* **Dual-Mono Detection:** The Center/Side guitar split algorithm actively checks for identical left/right channels. If a dual-mono track is detected, it halts the operation rather than generating silent or corrupted side channels.
* **Peak Normalization:** All successfully extracted stems are automatically peak-normalized (bringing the loudest transient to 0dB) to ensure consistent volume staging prior to DAW import.
