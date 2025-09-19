import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import multiprocessing
import logging
from datetime import datetime

def setup_logging(folder_path):
    """Set up logging to two files in the folder with datetime timestamp."""
    log_dir = Path(folder_path)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    renamed_handler = logging.FileHandler(log_dir / f'moved_{timestamp}.log')
    renamed_handler.setLevel(logging.INFO)
    renamed_formatter = logging.Formatter('%(asctime)s - %(message)s')
    renamed_handler.setFormatter(renamed_formatter)

    error_handler = logging.FileHandler(log_dir / f'move_errors_{timestamp}.log')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)

    renamed_logger = logging.getLogger('moved')
    renamed_logger.setLevel(logging.INFO)
    renamed_logger.addHandler(renamed_handler)

    error_logger = logging.getLogger('move_errors')
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)

    return renamed_logger, error_logger, timestamp

def is_media_file(file_path):
    """Check if file is a photo or video (add extensions as needed)."""
    media_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.gif', '.mov', '.mp4', '.avi', '.mkv', '.hevc', '.tif', '.tiff'}
    return Path(file_path).suffix.lower() in media_extensions

def get_creation_date(file_path):
    """Get the creation date using mdls on macOS. Fall back to today if not found."""
    try:
        result = subprocess.run(['mdls', '-name', 'kMDItemContentCreationDate', str(file_path)],
                                capture_output=True, text=True, check=True)
        if result.stdout.strip():
            # Parse output like "kMDItemContentCreationDate = 2025-09-19 10:30:45 +0000"
            date_part = result.stdout.split('=')[1].strip().strip('()')
            # Take up to timezone
            date_str = date_part.split('+')[0].strip()
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            return dt
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
    # Assume today if no date found
    return datetime.now()

def find_media_pairs(folder_path):
    """Walk the folder and find media files paired with their .xmp and .aae files."""
    folder = Path(folder_path)
    pairs = []  # List of (media_path, [list of sidecar_paths])

    for root, dirs, files in os.walk(folder):
        for file_name in files:
            file_path = Path(root) / file_name
            if is_media_file(file_path):
                stem = file_path.stem  # SHA256 hash
                sidecars = []

                # Check for .xmp
                xmp_path = file_path.parent / f"{stem}.xmp"
                if xmp_path.exists():
                    sidecars.append(xmp_path)

                # Check for .aae
                aae_path = file_path.parent / f"{stem}.aae"
                if aae_path.exists():
                    sidecars.append(aae_path)

                pairs.append((file_path, sidecars))

    return pairs

def get_date_task(pair, error_logger):
    """Task to get creation date for a media file."""
    media_file, sidecars = pair
    dt = get_creation_date(media_file)
    if dt is None:
        error_logger.error(f"Date retrieval failed for {media_file}. Sidecars: {[s.name for s in sidecars]}")
        return None
    return (media_file, sidecars, dt)

def compute_subfolder(dt, group_type):
    """Compute subfolder name based on group_type."""
    year = dt.year
    if group_type == 'monthly':
        month = dt.month
        return f"{year}M{month:02d}"
    elif group_type == 'weekly':
        week = dt.isocalendar().week
        return f"{year}W{week:02d}"
    elif group_type == 'daily':
        day = dt.timetuple().tm_yday
        return f"{year}D{day:03d}"
    elif group_type == 'quarterly':
        quarter = (dt.month - 1) // 3 + 1
        return f"{year}Q{quarter}"
    elif group_type == 'annually':
        return f"{year}"
    else:
        raise ValueError("Invalid group type")

def move_task(args, renamed_logger, error_logger, group_type):
    """Task to move a media file and its sidecars to the subfolder."""
    media_file, sidecars, dt = args

    # Compute subfolder
    try:
        subfolder_name = compute_subfolder(dt, group_type)
    except ValueError as e:
        error_logger.error(f"Invalid group type for {media_file}: {str(e)}")
        return False

    subfolder = media_file.parent / subfolder_name
    subfolder.mkdir(exist_ok=True)

    # New paths (same names in subfolder)
    new_media_path = subfolder / media_file.name
    new_sidecar_paths = [subfolder / s.name for s in sidecars]

    # Check for media collision
    if new_media_path.exists():
        error_logger.error(f"Move skipped for {media_file} due to collision in {subfolder_name}: {media_file.name} already exists. Sidecars: {[s.name for s in sidecars]}")
        return False

    # Move media
    try:
        media_file.rename(new_media_path)
        renamed_logger.info(f"Moved media to {subfolder_name}: {media_file.name}")
        success = True
    except OSError as e:
        error_logger.error(f"Move failed for media {media_file}: {str(e)}. Sidecars: {[s.name for s in sidecars]}")
        return False

    # Move sidecars if any
    sidecar_success = True
    for sidecar, new_path in zip(sidecars, new_sidecar_paths):
        if new_path.exists():
            error_logger.error(f"Sidecar move skipped for {sidecar}: {sidecar.name} already exists in {subfolder_name}.")
            sidecar_success = False
            continue

        try:
            sidecar.rename(new_path)
            renamed_logger.info(f"Moved sidecar to {subfolder_name}: {sidecar.name}")
        except OSError as e:
            error_logger.error(f"Sidecar move failed for {sidecar}: {str(e)}")
            sidecar_success = False

    return success and sidecar_success

def prompt_for_group_type():
    """Prompt user for grouping type."""
    options = {
        '1': 'monthly',   # YYYYMmm
        '2': 'weekly',    # YYYYWww
        '3': 'daily',     # YYYYDddd
        '4': 'quarterly', # YYYYQq
        '5': 'annually'   # YYYY
    }
    print("Choose grouping type:")
    print("1: Monthly (YYYYMmm, e.g., 2025M09)")
    print("2: Weekly (YYYYWww, e.g., 2025W37)")
    print("3: Daily (YYYYDddd, e.g., 2025D262)")
    print("4: Quarterly (YYYYQq, e.g., 2025Q3)")
    print("5: Annually (YYYY, e.g., 2025)")
    while True:
        choice = input("Enter number (1-5): ").strip()
        if choice in options:
            return options[choice]
        print("Invalid choice. Please enter 1-5.")

def organize_files(folder_path, num_threads, group_type):
    """Organize media files into subfolders by creation date, multithreaded."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Force Spotlight indexing to ensure metadata is available
    print("Forcing Spotlight indexing (this may take a moment)...")
    subprocess.run(['mdimport', str(folder)])

    # Setup logging
    renamed_logger, error_logger, timestamp = setup_logging(folder_path)

    # Find all pairs first
    print("Scanning for media files and XMP/AAE sidecar pairs...")
    pairs = find_media_pairs(folder_path)
    print(f"Found {len(pairs)} media file(s) to process.")

    if not pairs:
        print("No media files found.")
        return

    # Phase 1: Get dates in parallel
    print(f"Retrieving creation dates with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_pair = {executor.submit(get_date_task, pair, error_logger): pair for pair in pairs}
        date_args = []
        error_count = 0

        for future in as_completed(future_to_pair):
            result = future.result()
            if result is None:
                error_count += 1
            else:
                date_args.append(result)

    if error_count > 0:
        print(f"Warning: {error_count} files had date errors (check move_errors_{timestamp}.log).")

    if not date_args:
        print("No valid dates retrieved.")
        return

    # Phase 2: Move in parallel
    print(f"Moving files with {num_threads} threads...")
    moved_count = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_args = {executor.submit(move_task, args, renamed_logger, error_logger, group_type): args for args in date_args}

        for future in as_completed(future_to_args):
            if future.result():
                moved_count += 1

    print(f"\nSummary: Successfully moved {moved_count} files/pairs. Total processed: {len(date_args)}")
    print(f"Logs written to moved_{timestamp}.log and move_errors_{timestamp}.log in the folder.")

def prompt_for_threads():
    """Prompt user for number of threads, suggest based on cores."""
    try:
        cpu_count = multiprocessing.cpu_count()
    except ImportError:
        cpu_count = 8  # Fallback

    suggestion = min(cpu_count, 10)  # Cap at 10 for your M4
    default_input = input(f"Enter number of threads (default {suggestion}, max {cpu_count}): ").strip()
    if not default_input:
        num_threads = suggestion
    else:
        num_threads = int(default_input)
        if num_threads > cpu_count:
            print(f"Warning: Capping threads at {cpu_count} cores.")
            num_threads = cpu_count
    return num_threads

def prompt_for_folder():
    """Prompt user for the folder path."""
    while True:
        folder_input = input("Enter the path to the folder: ").strip()
        if not folder_input:
            print("Error: Path cannot be empty.")
            continue
        folder_path = Path(folder_input).expanduser()  # Expand ~ if used
        if folder_path.is_dir():
            return str(folder_path)
        else:
            print(f"Error: '{folder_input}' is not a valid directory. Please try again.")

if __name__ == "__main__":
    print("Media Organizer by Date Grouping")
    print("================================")

    folder_path = prompt_for_folder()
    print(f"Using folder: {folder_path}")

    group_type = prompt_for_group_type()
    print(f"Grouping by: {group_type}")

    num_threads = prompt_for_threads()
    print(f"Using {num_threads} threads.")

    organize_files(folder_path, num_threads, group_type)
