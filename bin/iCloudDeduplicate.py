import os
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import multiprocessing
import logging
from datetime import datetime
import re

def setup_logging(folder_path):
    """Set up logging to two files in the folder with datetime timestamp."""
    log_dir = Path(folder_path)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    renamed_handler = logging.FileHandler(log_dir / f'renamed_{timestamp}.log')
    renamed_handler.setLevel(logging.INFO)
    renamed_formatter = logging.Formatter('%(asctime)s - %(message)s')
    renamed_handler.setFormatter(renamed_formatter)
    
    error_handler = logging.FileHandler(log_dir / f'errors_{timestamp}.log')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)
    
    renamed_logger = logging.getLogger('renamed')
    renamed_logger.setLevel(logging.INFO)
    renamed_logger.addHandler(renamed_handler)
    
    error_logger = logging.getLogger('errors')
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)
    
    return renamed_logger, error_logger, timestamp

def get_sha256(file_path):
    """Compute SHA256 hash of the file using Python's hashlib."""
    try:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):  # Efficient chunking for large files
                sha256.update(chunk)
        return sha256.hexdigest()
    except (OSError, ValueError) as e:
        print(f"Error computing SHA256 for {file_path}: {e}")
        return None

def is_media_file(file_path):
    """Check if file is a photo or video (add extensions as needed)."""
    media_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.gif', '.mov', '.mp4', '.avi', '.mkv', '.hevc', '.tif', '.tiff'}
    return Path(file_path).suffix.lower() in media_extensions

def find_media_pairs(folder_path):
    """Walk the folder and find media files paired with their .xmp and .aae files (handling variant .aae naming)."""
    folder = Path(folder_path)
    pairs = []  # List of (media_path, [list of sidecar_paths])
    
    for root, dirs, files in os.walk(folder):
        file_set = set(files)  # Quick lookup for filenames in this dir
        
        for file_name in files:
            file_path = Path(root) / file_name
            if is_media_file(file_path):
                media_stem = Path(file_name).stem
                sidecars = []
                
                # Exact .xmp
                xmp_name = f"{media_stem}.xmp"
                if xmp_name in file_set:
                    sidecars.append(Path(root) / xmp_name)
                
                # Exact .aae
                aae_name = f"{media_stem}.aae"
                if aae_name in file_set:
                    sidecars.append(Path(root) / aae_name)
                
                # Appended 'O' .aae (e.g., IMG_7796 (1)O.aae)
                app_aae_name = f"{media_stem}O.aae"
                if app_aae_name in file_set:
                    sidecars.append(Path(root) / app_aae_name)
                
                # Inserted 'O' .aae (e.g., for IMG_1436 -> IMG_O1436.aae)
                match = re.search(r'^(.+)_(\d+)$', media_stem)
                if match:
                    prefix, digits = match.groups()
                    ins_aae_name = f"{prefix}_O{digits}.aae"
                    if ins_aae_name in file_set:
                        sidecars.append(Path(root) / ins_aae_name)
                
                pairs.append((file_path, sidecars))
    
    return pairs

def compute_hash_task(pair, error_logger):
    """Task to compute SHA256 for a media file."""
    media_file, sidecars = pair
    hash_value = get_sha256(media_file)
    if hash_value is None:
        error_logger.error(f"Hash computation failed for {media_file}. Sidecars: {[s.name for s in sidecars]}")
        return None
    return (media_file, sidecars, hash_value)

def rename_task(args, renamed_logger, error_logger):
    """Task to rename a media file and its XMP/AAE sidecars."""
    media_file, sidecars, hash_value = args
    
    # New names
    new_media_name = f"{hash_value}{media_file.suffix}"
    new_media_path = media_file.parent / new_media_name
    
    # Check for media collision
    if new_media_path.exists():
        error_logger.error(f"Rename skipped for {media_file} due to name collision: {new_media_name} already exists. Sidecars: {[s.name for s in sidecars]}")
        return False
    
    # Rename media
    try:
        media_file.rename(new_media_path)
        renamed_logger.info(f"Renamed media: {media_file.name} -> {new_media_name}")
        success = True
    except OSError as e:
        error_logger.error(f"Rename failed for media {media_file}: {str(e)}. Sidecars: {[s.name for s in sidecars]}")
        return False
    
    # Rename sidecars if any
    sidecar_success = True
    for sidecar in sidecars:
        new_sidecar_name = f"{hash_value}{sidecar.suffix}"
        new_sidecar_path = media_file.parent / new_sidecar_name
        
        # Check for sidecar collision
        if new_sidecar_path.exists():
            error_logger.error(f"Sidecar rename skipped for {sidecar}: {new_sidecar_name} already exists.")
            sidecar_success = False
            continue
        
        try:
            sidecar.rename(new_sidecar_path)
            renamed_logger.info(f"Renamed sidecar: {sidecar.name} -> {new_sidecar_name}")
        except OSError as e:
            error_logger.error(f"Sidecar rename failed for {sidecar}: {str(e)}")
            sidecar_success = False
    
    return success and sidecar_success

def rename_with_sha256(folder_path, num_threads):
    """Rename media files and their .xmp/.aae sidecars in the folder using SHA256, multithreaded."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    # Setup logging
    renamed_logger, error_logger, timestamp = setup_logging(folder_path)
    
    # Find all pairs first
    print("Scanning for media files and XMP/AAE sidecar pairs...")
    pairs = find_media_pairs(folder_path)
    print(f"Found {len(pairs)} media file(s) to process.")
    
    if not pairs:
        print("No media files found.")
        return
    
    # Phase 1: Compute hashes in parallel
    print(f"Computing SHA256 hashes with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_pair = {executor.submit(compute_hash_task, pair, error_logger): pair for pair in pairs}
        hash_args = []
        error_count = 0
        
        for future in as_completed(future_to_pair):
            result = future.result()
            if result is None:
                error_count += 1
            else:
                hash_args.append(result)
    
    if error_count > 0:
        print(f"Warning: {error_count} files had hash errors (check errors_{timestamp}.log).")
    
    if not hash_args:
        print("No valid hashes computed.")
        return
    
    # Phase 2: Rename in parallel
    print(f"Renaming with {num_threads} threads...")
    renamed_count = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_args = {executor.submit(rename_task, args, renamed_logger, error_logger): args for args in hash_args}
        
        for future in as_completed(future_to_args):
            if future.result():
                renamed_count += 1
    
    print(f"\nSummary: Successfully renamed {renamed_count} files/pairs. Total processed: {len(hash_args)}")
    print(f"Logs written to renamed_{timestamp}.log and errors_{timestamp}.log in the folder.")

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
    print("SHA256 File Renamer for Media, XMP, and AAE Files")
    print("================================================")
    
    folder_path = prompt_for_folder()
    print(f"Using folder: {folder_path}")
    
    num_threads = prompt_for_threads()
    print(f"Using {num_threads} threads.")
    
    rename_with_sha256(folder_path, num_threads)
