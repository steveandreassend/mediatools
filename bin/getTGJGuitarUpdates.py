#!/usr/bin/env python3

from playwright.sync_api import sync_playwright
from datetime import datetime
from pathlib import Path
import html


###############################################################################
# SONGS
###############################################################################

SONGS = [
    {"title": "99 LUFTBALLONS", "artist": "NENA", "url": "https://my.hidrive.com/share/3-.8ipp5r8"},
    {"title": "ALLEIN ALLEIN", "artist": "POLARKREIS 18", "url": "https://my.hidrive.com/share/qp121e097c"},
    {"title": "ALL I WANT FOR CHRISTMAS", "artist": "MARIAH CAREY", "url": "https://my.hidrive.com/share/hzka9zn1jd"},
    {"title": "ALT WIE EIN BAUM", "artist": "PUHDYS", "url": "https://my.hidrive.com/share/atttuz2lyp"},
    {"title": "ANGELS", "artist": "ROBBIE WILLIAMS", "url": "https://my.hidrive.com/share/143pjxk8fv"},
    {"title": "AN GUTEN TAGEN", "artist": "JOHANNES OERDING", "url": "https://my.hidrive.com/share/x7ifb3oio7"},
    {"title": "BACK IN TIME", "artist": "HUEY LEWIS & THE NEWS", "url": "https://my.hidrive.com/share/dx3nhk1ayk"},
    {"title": "BEAUTIFUL DAY", "artist": "U2", "url": "https://my.hidrive.com/share/6ige-s-bta"},
    {"title": "BLINDING LIGHTS", "artist": "THE WEEKND", "url": "https://my.hidrive.com/share/rsxkvuvgfn"},
    {"title": "CAN'T STOP", "artist": "RED HOT CHILI PEPPERS", "url": "https://my.hidrive.com/share/0sbe53opnt"},
    {"title": "COUNTRY ROADS", "artist": "JOHN DENVER", "url": "https://my.hidrive.com/share/wws1pyu6gr"},
    {"title": "DANCING QUEEN", "artist": "ABBA", "url": "https://my.hidrive.com/share/q7lv-ki140"},
    {"title": "DIE HESSE KOMME", "artist": "RODGAU MONOTONES", "url": "https://my.hidrive.com/share/to3we42y95"},
    {"title": "DON'T GO BREAKING MY HEART", "artist": "ELTON JOHN", "url": "https://my.hidrive.com/share/ql56r4yqmf"},
    {"title": "DON'T LOOK BACK IN ANGER", "artist": "OASIS", "url": "https://my.hidrive.com/share/nlp7k57a1e"},
    {"title": "DON'T STOP BELIEVIN'", "artist": "JOURNEY", "url": "https://my.hidrive.com/share/xv9moag1d8"},
    {"title": "DON'T STOP ME NOW", "artist": "QUEEN", "url": "https://my.hidrive.com/share/zl9cybiuyl"},
    {"title": "DOPAMINE", "artist": "PURPLE DISCO MACHINE", "url": "https://my.hidrive.com/share/hk15gzv684"},
    {"title": "DREAMER", "artist": "OZZY OSBOURNE", "url": "https://my.hidrive.com/share/pk.nq44s9d"},
    {"title": "DYNAMITE", "artist": "BTS", "url": "https://my.hidrive.com/share/hecenr81cv"},
    {"title": "EVERYBODY", "artist": "BACKSTREET BOYS", "url": "https://my.hidrive.com/share/yubmyw2bqf"},
    {"title": "FLASHDANCE WHAT A FEELING", "artist": "IRENE CARA", "url": "https://my.hidrive.com/share/n954v3ugee"},
    {"title": "HALLELUJAH", "artist": "LEONARD COHEN", "url": "https://my.hidrive.com/share/xqoq0u174i"},
    {"title": "HUNGRY HEART", "artist": "BRUCE SPRINGSTEEN", "url": "https://my.hidrive.com/share/3fdi2hpazt"},
    {"title": "ITS RAINING AGAIN", "artist": "SUPERTRAMP", "url": "https://my.hidrive.com/share/mtkt8jo7si"},
    {"title": "ITS MY LIFE", "artist": "BON JOVI", "url": "https://my.hidrive.com/share/ul4s.3ph04"},
    {"title": "JUST CAN'T GET ENOUGH", "artist": "DEPECHE MODE", "url": "https://my.hidrive.com/share/meucszfbq6"},
    {"title": "LADIES MASHUP", "artist": "MADONNA / BEYONCE / HELENE FISCHER", "url": "https://my.hidrive.com/share/v3hcnkdmqc"},
    {"title": "LAST CHRISTMAS", "artist": "WHAM", "url": "https://my.hidrive.com/share/c7323uae91"},
    {"title": "LET IT BE", "artist": "THE BEATLES", "url": "https://my.hidrive.com/share/ru24b8t.sy"},
    {"title": "LET IT GO", "artist": "FROZEN", "url": "https://my.hidrive.com/share/qeu1fn1nzd"},
    {"title": "LET US ENTERTAIN YOU", "artist": "ROBBIE WILLIAMS", "url": "https://my.hidrive.com/share/y5gtc9gbg7"},
    {"title": "LOVE YOURSELF", "artist": "JUSTIN BIEBER", "url": "https://my.hidrive.com/share/xs2m-b2stm"},
    {"title": "MR. BRIGHTSIDE", "artist": "THE KILLERS", "url": "https://my.hidrive.com/share/iejain2ens"},
    {"title": "MUSIC", "artist": "JOHN MILES", "url": "https://my.hidrive.com/share/1wbc12np68"},
    {"title": "NARCOTIC", "artist": "LIQUIDO", "url": "https://my.hidrive.com/share/y.f.yjn4dp"},
    {"title": "NIE ZUVOR", "artist": "ELECTRA", "url": "https://my.hidrive.com/share/1l4oopx41c"},
    {"title": "RADIO GAGA", "artist": "QUEEN", "url": "https://my.hidrive.com/share/6x0gklh-rk"},
    {"title": "SCHWARZ WEISS WIE SCHNEE", "artist": "TANKARD", "url": "https://my.hidrive.com/share/51opbz70ls"},
    {"title": "SEPTEMBER", "artist": "EARTH WIND & FIRE", "url": "https://my.hidrive.com/share/r5tki9unax"},
    {"title": "SHAKE IT OFF", "artist": "TAYLOR SWIFT", "url": "https://my.hidrive.com/share/x3g7n54.ov"},
    {"title": "SHUT UP AND DANCE", "artist": "WALK THE MOON", "url": "https://my.hidrive.com/share/7u0lgq7n9p"},
    {"title": "SIMPLY THE BEST", "artist": "TINA TURNER", "url": "https://my.hidrive.com/share/bc1ur1csez"},
    {"title": "SMELLS LIKE TEEN SPIRIT", "artist": "NIRVANA", "url": "https://my.hidrive.com/share/23g4qs6c1q"},
    {"title": "STÄÄNE", "artist": "BAP", "url": "https://my.hidrive.com/share/idmix671j2"},
    {"title": "START ME UP", "artist": "ROLLING STONES", "url": "https://my.hidrive.com/share/izns98mgay"},
    {"title": "SUMMER OF 69", "artist": "BRYAN ADAMS", "url": "https://my.hidrive.com/share/y374tffbzr"},
    {"title": "SWEET CAROLINE", "artist": "NEIL DIAMOND", "url": "https://my.hidrive.com/share/jywilisiyc"},
    {"title": "SWEET CHILD O' MINE", "artist": "GUNS N' ROSES", "url": "https://my.hidrive.com/share/2ksjz-32ck"},
    {"title": "TAGE WIE DIESE", "artist": "DIE TOTEN HOSEN", "url": "https://my.hidrive.com/share/p0gssriuip"},
    {"title": "THAT'S THE WAY IT IS", "artist": "CELINE DION", "url": "https://my.hidrive.com/share/pnvnl023kf"},
    {"title": "THE EMPTINESS MACHINE", "artist": "LINKIN PARK", "url": "https://my.hidrive.com/share/5srzv6o26i"},
    {"title": "THE FINAL COUNTDOWN", "artist": "EUROPE", "url": "https://my.hidrive.com/share/zirf4gthtp"},
    {"title": "TITANIUM", "artist": "DAVID GUETTA / SIA", "url": "https://my.hidrive.com/share/nj90wuh6-p"},
    {"title": "UPTOWN FUNK", "artist": "MARK RONSON / BRUNO MARS", "url": "https://my.hidrive.com/share/9b.56948g4"},
    {"title": "VERDAMP LANG HER", "artist": "BAP", "url": "https://my.hidrive.com/share/j3apfj9x1l"},
    {"title": "VIVA LA VIDA", "artist": "COLDPLAY", "url": "https://my.hidrive.com/share/d9qkk39z6l"},
    {"title": "WE ARE FAMILY", "artist": "SISTER SLEDGE", "url": "https://my.hidrive.com/share/1gwqt8p79o"},
    {"title": "WE ARE THE WORLD", "artist": "USA FOR AFRICA", "url": "https://my.hidrive.com/share/oo17gi1y7o"},
    {"title": "WE WILL ROCK YOU", "artist": "QUEEN", "url": "https://my.hidrive.com/share/rh1fnc7uon"},
    {"title": "ZOMBIE", "artist": "THE CRANBERRIES", "url": "https://my.hidrive.com/share/4jwd0vo7c5"}
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

        page.goto(
            song["url"],
            wait_until="networkidle",
            timeout=60000
        )

        page.wait_for_timeout(1500)

        if not dir_data:
            return {
                **song,
                "newest_pdf": "No directory data found",
                "modified": "N/A",
                "mtime": 0,
            }

        members = dir_data.get("members", [])

        pdfs = [
            f for f in members
            if f.get("name", "").lower().endswith(".pdf")
        ]

        if not pdfs:
            return {
                **song,
                "newest_pdf": "No PDF found",
                "modified": "N/A",
                "mtime": 0,
            }

        newest_pdf = max(
            pdfs,
            key=lambda f: f.get("mtime", 0)
        )

        modified_dt = datetime.fromtimestamp(
            newest_pdf["mtime"]
        )

        return {
            **song,
            "newest_pdf": newest_pdf["name"],
            "modified": modified_dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "mtime": newest_pdf["mtime"],
        }

    except Exception as exc:

        return {
            **song,
            "newest_pdf": f"ERROR: {exc}",
            "modified": "N/A",
            "mtime": 0,
        }

    finally:
        page.close()


###############################################################################
# GENERATE HTML REPORT
###############################################################################

def generate_html_report(results):

    now = datetime.now()

    report_timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    filename = (
        "hidrive_report_"
        + now.strftime("%Y%m%d_%H%M%S")
        + ".html"
    )

    rows = []

    for r in results:

        age_days_display = ""
        age_sort = 99999

        if r["mtime"] > 0:

            modified_dt = datetime.fromtimestamp(
                r["mtime"]
            )

            age_days = (
                datetime.now() - modified_dt
            ).days

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

        rows.append(
            f"""
<tr class="{css_class}">
    <td>
        <a href="{html.escape(r['url'])}" target="_blank">
            {html.escape(r['title'])}
        </a>
    </td>

    <td>{html.escape(r['artist'])}</td>

    <td>{html.escape(r['newest_pdf'])}</td>

    <td data-sort="{date_sort}">
        {html.escape(r['modified'])}
    </td>

    <td data-sort="{age_sort}">
        {age_days_display}
    </td>
</tr>
"""
        )

    html_content = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="utf-8">

<title>HiDrive PDF Report</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 20px;
}}

h1 {{
    margin-bottom: 5px;
}}

.timestamp {{
    color: #555;
    margin-bottom: 20px;
}}

.legend {{
    margin-bottom: 20px;
}}

.legend span {{
    display: inline-block;
    padding: 6px 12px;
    margin-right: 10px;
    border-radius: 4px;
}}

.red {{
    background-color: #ffb3b3;
}}

.yellow {{
    background-color: #fff3b0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background-color: #2c3e50;
    color: white;
    padding: 10px;
    text-align: left;
    cursor: pointer;
}}

td {{
    border: 1px solid #ddd;
    padding: 8px;
}}

tr:nth-child(even) {{
    background-color: #f5f5f5;
}}

.recent {{
    background-color: #ffb3b3 !important;
}}

.warning {{
    background-color: #fff3b0 !important;
}}

a {{
    color: #0066cc;
    text-decoration: none;
    font-weight: bold;
}}

a:hover {{
    text-decoration: underline;
}}

th:hover {{
    background-color: #34495e;
}}

</style>

<script>

const sortDirection = {{}};

function sortTable(columnIndex)
{{
    var table = document.getElementById("songTable");
    var tbody = table.tBodies[0];

    var rows = Array.from(tbody.rows);

    sortDirection[columnIndex] =
        !sortDirection[columnIndex];

    const ascending =
        sortDirection[columnIndex];

    rows.sort(function(a, b)
    {{
        let av =
            a.cells[columnIndex].dataset.sort ||
            a.cells[columnIndex].innerText.trim();

        let bv =
            b.cells[columnIndex].dataset.sort ||
            b.cells[columnIndex].innerText.trim();

        const an = parseFloat(av);
        const bn = parseFloat(bv);

        if (!isNaN(an) && !isNaN(bn))
        {{
            return ascending
                ? an - bn
                : bn - an;
        }}

        return ascending
            ? av.localeCompare(bv)
            : bv.localeCompare(av);
    }});

    rows.forEach(row =>
        tbody.appendChild(row)
    );
}}

window.onload = function()
{{
    sortTable(0);
}};

</script>

</head>

<body>

<h1>HiDrive Song PDF Report</h1>

<div class="timestamp">
<b>Last Poll Run:</b> {report_timestamp}
</div>

<div class="legend">
    <span class="red">
        PDF updated within last 30 days
    </span>

    <span class="yellow">
        PDF updated within last 90 days
    </span>
</div>

<table id="songTable">

<thead>
<tr>
    <th onclick="sortTable(0)">Song</th>
    <th onclick="sortTable(1)">Artist</th>
    <th onclick="sortTable(2)">Newest PDF</th>
    <th onclick="sortTable(3)">Modified Date</th>
    <th onclick="sortTable(4)">Age (Days)</th>
</tr>
</thead>

<tbody>
{''.join(rows)}
</tbody>

</table>

</body>
</html>
"""

    Path(filename).write_text(
        html_content,
        encoding="utf-8"
    )

    return filename


###############################################################################
# MAIN
###############################################################################

def main():

    total = len(SONGS)

    print(
        f"Scanning {total} HiDrive folders...\n",
        flush=True
    )

    results = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        for index, song in enumerate(
            SONGS,
            start=1
        ):

            print(
                f"[{index}/{total}] "
                f"{song['title']} - {song['artist']}",
                flush=True
            )

            result = scan_song(
                browser,
                song
            )

            results.append(result)

            print(
                f"    PDF : {result['newest_pdf']}",
                flush=True
            )

            print(
                f"    Date: {result['modified']}\n",
                flush=True
            )

        browser.close()

    results.sort(
        key=lambda r: r["title"].lower()
    )

    report_file = generate_html_report(
        results
    )

    print()
    print("Scan complete.")
    print(f"HTML report written to: {report_file}")
    print()


if __name__ == "__main__":
    main()
