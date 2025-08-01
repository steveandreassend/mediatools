# Holiday Media Enhancement Scripts

This repository contains Python and shell scripts to enhance your holiday photos, videos, and online content. Below is a description of each script and pro tips for their use.

## Scripts

### coloringPicture.py
Converts any holiday photo into a printable coloring-in picture for kids, allowing them to relive memories creatively.

- **Pro Tip 1**: Print the coloring-in picture with the original photo in the corner to guide kids on color choices.
- **Pro Tip 2**: For imaginary themes, use Grok (not ChatGPT or Gemini) to generate a photo of the desired scene, then process it with this script for printing.

### imageUpscaler.py
Upscales holiday photos to 4K or 8K resolution using a deep learning neural network to fill gaps and enhance faces, delivering sharp, high-quality results even on older iPhone devices.

- **Features**: Increases resolution, enhances faces, and applies additional quality-improving techniques, even for some blurry photos.
- **Benefit**: Zoom in without losing clarity, potentially eliminating the need for a new high-end camera phone.
- **Requirements**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use. CPU mode is available but significantly slower.

### videoUpscaler.py
Upscales holiday videos frame by frame to improve quality, with batch processing for efficiency.

- **Requirements**: A fast laptop with a GPU (e.g., MacBook Pro M4 with integrated GPU or a system with an Nvidia GPU) is recommended for practical use. CPU mode is available but significantly slower.
- **Warning**: Running on a basic laptop in CPU mode may result in long processing times.

### videoDownloader.sh
Downloads any online video in the highest quality as a portable MP4 file for offline viewing on any device.

- **Features**: No ads, cookies, mobile data, or tracking, ensuring a clean and private experience.

### transcribeVideo.py
Generates transcripts for videos (e.g., YouTube seminars) by checking for existing transcripts or analyzing audio if none are available.

- **Use Case**: Save time by summarizing long videos. You can use your preferred AI tool to format, paraphrase, or create an executive summary of the transcript, or have it read aloud while driving.
- **Pro Tip**: For multiple or serial seminars, combine transcripts or summaries into one document and use an AI chat to explore common themes and intersectionality for deeper insights.
