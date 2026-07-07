import re
import os
from bs4 import BeautifulSoup

def extract_songs_from_html(html_file_path):
    """
    Extracts song titles, artists, and HiDrive links from the Grand Jam tutorial HTML.
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Find all the tab navigation items (these contain the Song Titles and Artists)
    tab_items = soup.find_all('li', class_='tab-item')

    # 2. Find all the HiDrive links in the document
    hidrive_links = soup.find_all('a', href=re.compile(r'https://my\.hidrive\.com/share/.*'))

    songs_array = []

    # Match titles to links.
    for i, tab in enumerate(tab_items):
        title_artist_span = tab.find('span')
        if title_artist_span:
            raw_text = title_artist_span.get_text(strip=True)

            # Split "SONG TITLE - ARTIST" or "SONG TITLE – ARTIST"
            parts = raw_text.split(' - ', 1)
            if len(parts) == 1:
                parts = raw_text.split(' – ', 1)

            title = parts[0].strip() if len(parts) > 0 else raw_text
            artist = parts[1].strip() if len(parts) > 1 else ""

            url = ""
            if i < len(hidrive_links):
                url = hidrive_links[i].get('href')

            songs_array.append({
                "title": title,
                "artist": artist,
                "url": url
            })

    return songs_array

if __name__ == "__main__":
    # Prompt for the specific instrument's HTML file
    filename = input("Enter the HTML file name for the instrument (e.g., bass.html, guitar.html): ").strip()

    if not os.path.exists(filename):
        print(f"\nError: The file '{filename}' was not found in the current directory.")
        print("Please ensure the file is saved in the exact same folder as this script.")
    else:
        print(f"\nExtracting data from {filename}...")
        songs = extract_songs_from_html(filename)

        # Determine the output filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_filename = f"{base_name}.py"

        # Write the array to the output file
        with open(output_filename, 'w', encoding='utf-8') as out_file:
            out_file.write("SONGS = [\n")
            for song in songs:
                out_file.write(f'    {{"title": "{song["title"]}", "artist": "{song["artist"]}", "url": "{song["url"]}"}},\n')
            out_file.write("]\n")

        print(f"Successfully generated {output_filename} containing {len(songs)} songs.")
