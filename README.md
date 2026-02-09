# Media Tools Repo

This repository contains Python and shell scripts to run on your laptop that:
* Enhance your holiday photos and videos. Upscale to 4K or 8K resolution. Correct blurry images. Fix closed eyes.
* Create coloring-in pictures from your photo memories that you can print out.
* Download videos from websites and save in high quality video and audio.
* Transcribe and summarize content (videos, audio, websites, docs) that is online or offline to make executive summaries using your chosen LLM running in Ollama.
* Shrink the size of your iCloud camera roll by archiving your photos and videos to USB. Free up space on your phone and iCloud.
* Examine the frequency of recordings when converting from 440Hz to 432Hz.
* Generate 3D sound to replicate Dolby Atmos effects by combining audio tracks.

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

### transcribeVideo.py
Generates transcripts for videos (e.g., YouTube seminars) by checking for existing transcripts or analyzing audio if none are available. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing videos of long talks or seminars. You can use your preferred AI tool to format, paraphrase, or create an executive summary of the transcript, or have it read aloud while driving.
- **Pro Tip**: For multiple or serial seminars, combine transcripts or summaries into one document and use an AI chat to explore common themes and intersectionality for deeper insights.

### transcribeSummarize.py
Generates an executive summary of an online video using Ollama with Meta's Llama3 model for summarization, and the YouTube Transcript API or Google's Web Speech API for transcription. The same method as transcribeVideo.py is used to obtain the transcript. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing long videos with an executive summary with a list of the key takeaways - PRIVATELY on your local Mac, ACCURATELY without consulting external sources, and for FREE without requiring a subscription fee. It covers the situations where ChatGPT (et al.) either i) refuses to make a summary of a video when there are no transcripts on the video, or ii) if it does, it peppers the summary with information from external sources that may distort the output.
- **Pro Tip 1**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use and at least 24GB of system memory.
- **Pro Tip 2**: The Ollama server must be running with the Llama3 model. Start it with 'ollama serve'.
- **Pro Tip 3**: Adapt the code if you prefer to use a remote cloud-based LLM to perform the document summarization.
- **Pro Tip 4**: You can substantiate Llama3 with the model of your choice.

### summarizeDocument.py
Generates an executive summary of a document using Ollama with Meta's Llama3 model for summarization. Works with Word docx, PDF, or plain text files. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing documents with an executive summary with a list of the key takeaways, again with the same benefits as transcribeSummarize.py.
- **Pro Tip**: Use the script benchmarkSummarization.sh to run a stress test on MacOS of the summarizeDocument.py performance.

### summarizePage.py
Generates an executive summary of a webpage using Ollama with Meta's Llama3 model for summarization. Requires python 3.10 or later.

- **Use Case**: Save time by summarizing webpages with an executive summary with a list of the key takeaways, again with the same benefits as transcribeSummarize.py.

### summarizeAudio.py
Generates an executive summary of an audio file using Ollama with Meta's Llama3 model for summarization. Requires python 3.10 or later.

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
