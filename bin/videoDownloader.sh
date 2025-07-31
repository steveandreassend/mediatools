#!/bin/bash

# Prompt user for the YouTube URL
read -p "Enter the Video URL: " VIDEO_URL

# Define the output file name based on the video ID or any other logic
OUTPUT_DIR="./"  # You can change this to any folder you prefer
INPUTFILE="$OUTPUT_DIR/$(yt-dlp --get-filename -o "%(id)s.%(ext)s" "$VIDEO_URL")"

# Run yt-dlp to download the video (restrict filenames and select format) in highest quality
yt-dlp --restrict-filenames -f "bv*+ba/b" "$VIDEO_URL" -o "$INPUTFILE"

# Check if yt-dlp successfully downloaded the file
if [ ! -f "$INPUTFILE" ]; then
    echo "Error: Failed to download the video."
    exit 1
fi

# Define the output file for the conversion
OUTPUTFILE="${INPUTFILE%.webm}.mp4"

# Run ffmpeg to convert the video to mp4 format
ffmpeg -i "$INPUTFILE" -c:v libx264 -crf 23 -preset fast "$OUTPUTFILE"

# Check if ffmpeg successfully created the output file
if [ -f "$OUTPUTFILE" ]; then
    echo "Conversion successful! The video is saved as $OUTPUTFILE"
else
    echo "Error: Failed to convert the video."
    exit 1
fi
