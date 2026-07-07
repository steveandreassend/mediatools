from playwright.sync_api import sync_playwright
from datetime import datetime
from pathlib import Path
import html
import os
import sys
import importlib.util

# Define the default list of instruments based on your files
DEFAULT_INSTRUMENTS = [
    "bass",
    "brass_sax",
    "drums",
    "strings",
    "guitar",
    "keyboard",
    "vocals"
]

###############################################################################
# SCAN A SINGLE HIDRIVE FOLDER
###############################################################################

def scan_song(browser, song):
    dir_data = None

    def handle_response(response):
        nonlocal dir_data
        try:
            if "/api/dir?" in response.url:
                dir_data = response.json()
        except Exception:
            pass

    page = browser.new_page()
    page.on("response", handle_response)

    try:
        page.goto(song["url"], wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)

        if not dir_data:
            return {**song, "newest_pdf": "No directory data found", "modified": "N/A", "mtime": 0}

        members = dir_data.get("members", [])
        pdfs = [f for f in members if f.get("name", "").lower().endswith(".pdf")]

        if not pdfs:
            return {**song, "newest_pdf": "No PDF found", "modified": "N/A", "mtime": 0}

        newest_pdf = max(pdfs, key=lambda f: f.get("mtime", 0))
        modified_dt = datetime.fromtimestamp(newest_pdf["mtime"])

        return {
            **song,
            "newest_pdf": newest_pdf["name"],
            "modified": modified_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "mtime": newest_pdf["mtime"],
        }
    except Exception as exc:
        return {**song, "newest_pdf": f"ERROR: {exc}", "modified": "N/A", "mtime": 0}
    finally:
        page.close()

###############################################################################
# GENERATE HTML CONTENT
###############################################################################

def generate_html_content(all_results):
    now = datetime.now()
    report_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Generate Table of Contents
    toc_html = "<h2>Table of Contents</h2><ul>"
    for inst in all_results.keys():
        display_name = inst.replace("_", " ").title()
        toc_html += f'<li><a href="#section-{inst}">{display_name}</a></li>'
    toc_html += "</ul><hr>"

    # Generate Subsections
    sections_html = ""
    for inst, results in all_results.items():
        display_name = inst.replace("_", " ").title()

        # Pre-sort results by Age (Days) ascending (mtime descending, but 0 at the bottom)
        # This ensures PDFs look correct without needing JS to run
        results.sort(key=lambda r: (r['mtime'] == 0, -r['mtime']))

        rows = []
        for r in results:
            age_days_display = ""
            age_sort = 99999

            if r["mtime"] > 0:
                modified_dt = datetime.fromtimestamp(r["mtime"])
                age_days = (now - modified_dt).days
                age_days_display = str(age_days)
                age_sort = age_days

                if age_days <= 30:
                    css_class = "recent"
                elif age_days <= 90:
                    css_class = "warning"
                else:
                    css_class = ""
                date_sort = r["mtime"]
            else:
                css_class = ""
                date_sort = 0

            rows.append(f"""
            <tr class="{css_class}">
                <td><a href="{html.escape(r['url'])}" target="_blank">{html.escape(r['title'])}</a></td>
                <td>{html.escape(r['artist'])}</td>
                <td>{html.escape(r['newest_pdf'])}</td>
                <td data-sort="{date_sort}">{html.escape(r['modified'])}</td>
                <td data-sort="{age_sort}">{age_days_display}</td>
            </tr>
            """)

        sections_html += f"""
        <h2 id="section-{inst}">{display_name}</h2>
        <table id="table-{inst}">
            <thead>
                <tr>
                    <th onclick="sortTable('table-{inst}', 0)">Song</th>
                    <th onclick="sortTable('table-{inst}', 1)">Artist</th>
                    <th onclick="sortTable('table-{inst}', 2)">Newest PDF</th>
                    <th onclick="sortTable('table-{inst}', 3)">Modified Date</th>
                    <th onclick="sortTable('table-{inst}', 4)">Age (Days)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <br><br>
        """

    html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HiDrive Combined Report</title>
<style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 20px; }}
    h1, h2 {{ margin-bottom: 5px; color: #2c3e50; }}
    .timestamp {{ color: #555; margin-bottom: 20px; font-weight: bold; }}
    .legend {{ margin-bottom: 20px; }}
    .legend span {{ display: inline-block; padding: 6px 12px; margin-right: 10px; border-radius: 4px; border: 1px solid #ccc; }}
    .red {{ background-color: #ffb3b3; }}
    .yellow {{ background-color: #fff3b0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; page-break-inside: auto; }}
    tr {{ page-break-inside: avoid; page-break-after: auto; }}
    th {{ background-color: #2c3e50; color: white; padding: 10px; text-align: left; cursor: pointer; }}
    td {{ border: 1px solid #ddd; padding: 8px; }}
    tr:nth-child(even) {{ background-color: #f5f5f5; }}
    .recent {{ background-color: #ffb3b3 !important; }}
    .warning {{ background-color: #fff3b0 !important; }}
    a {{ color: #0066cc; text-decoration: none; font-weight: bold; }}
    a:hover {{ text-decoration: underline; }}
    th:hover {{ background-color: #34495e; }}
    hr {{ border: 0; height: 1px; background: #ddd; margin: 20px 0; }}
    ul {{ line-height: 1.6; }}
    @media print {{
        a {{ text-decoration: none; color: black; }}
        body {{ margin: 0; }}
        table {{ font-size: 10pt; }}
    }}
</style>
<script>
    const sortDirections = {{}};
    function sortTable(tableId, columnIndex) {{
        var table = document.getElementById(tableId);
        var tbody = table.tBodies[0];
        var rows = Array.from(tbody.rows);

        if (!sortDirections[tableId]) sortDirections[tableId] = {{}};
        sortDirections[tableId][columnIndex] = !sortDirections[tableId][columnIndex];
        const ascending = sortDirections[tableId][columnIndex];

        rows.sort(function(a, b) {{
            let av = a.cells[columnIndex].dataset.sort || a.cells[columnIndex].innerText.trim();
            let bv = b.cells[columnIndex].dataset.sort || b.cells[columnIndex].innerText.trim();
            const an = parseFloat(av);
            const bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return ascending ? an - bn : bn - an;
            return ascending ? av.localeCompare(bv) : bv.localeCompare(av);
        }});
        rows.forEach(row => tbody.appendChild(row));
    }}
</script>
</head>
<body>

<h1>The Grand Jam - Combined Updates Report</h1>
<div class="timestamp">Last Poll Run: {report_timestamp}</div>

<div class="legend">
    <span class="red">PDF updated within last 30 days</span>
    <span class="yellow">PDF updated within last 90 days</span>
</div>

{toc_html}
{sections_html}

</body>
</html>
"""
    return html_template

###############################################################################
# LOAD SONGS DYNAMICALLY
###############################################################################

def load_songs_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: The file '{filepath}' was not found. Skipping.")
        return None

    try:
        spec = importlib.util.spec_from_file_location("songs_module", filepath)
        songs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(songs_module)

        if hasattr(songs_module, 'SONGS'):
            return songs_module.SONGS
        else:
            print(f"Warning: '{filepath}' does not contain a 'SONGS' variable. Skipping.")
            return None
    except Exception as e:
        print(f"Error loading file '{filepath}': {e}")
        return None

###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    print("=== HiDrive Combined Updates Scanner ===")

    # 1. Prompt for instruments
    default_inst_str = ", ".join(DEFAULT_INSTRUMENTS)
    inst_input = input(f"\nEnter instruments to scan (comma-separated)\n[Default: {default_inst_str}]: ").strip()

    if not inst_input:
        selected_instruments = DEFAULT_INSTRUMENTS
    else:
        selected_instruments = [i.strip().lower() for i in inst_input.split(",")]

    # 2. Prompt for output format
    fmt_input = input("Output format (HTML or PDF) [Default: HTML]: ").strip().upper()
    if fmt_input not in ["HTML", "PDF"]:
        fmt_input = "HTML"

    all_results = {}

    print(f"\nInitializing Playwright to scan {len(selected_instruments)} sections...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for inst in selected_instruments:
            filepath = f"{inst}.py"
            print(f"\n--- Loading {filepath} ---")

            SONGS = load_songs_from_file(filepath)
            if not SONGS:
                continue

            total = len(SONGS)
            print(f"Scanning {total} folders for {inst}...")

            inst_results = []
            for index, song in enumerate(SONGS, start=1):
                print(f"  [{index}/{total}] {song['title']} - {song['artist']}", flush=True)

                result = scan_song(browser, song)
                inst_results.append(result)

            all_results[inst] = inst_results

        # Ensure we have data before rendering
        if not all_results:
            print("\nNo data was successfully retrieved. Exiting.")
            browser.close()
            sys.exit(1)

        # Generate the HTML payload
        print("\nGenerating report content...")
        html_content = generate_html_content(all_results)

        # Save mechanism
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if fmt_input == "HTML":
            out_file = f"hidrive_combined_{timestamp_str}.html"
            Path(out_file).write_text(html_content, encoding="utf-8")
            print(f"\n✅ HTML report written to: {out_file}")

        elif fmt_input == "PDF":
            out_file = f"hidrive_combined_{timestamp_str}.pdf"
            print(f"Rendering PDF in landscape mode to {out_file} (this may take a few seconds)...")

            # Use Playwright to render the HTML as a landscape PDF
            pdf_page = browser.new_page()
            pdf_page.set_content(html_content, wait_until="networkidle")

            pdf_page.pdf(
                path=out_file,
                format="A4",
                landscape=True,
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"}
            )
            print(f"\n✅ PDF report written to: {out_file}")

        browser.close()

if __name__ == "__main__":
    main()
