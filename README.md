# Holiday Media Enhancement Scripts

This repository contains Python and shell scripts to enhance your holiday photos, videos, and online content. Below is a description of each script and pro tips for their use.

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
