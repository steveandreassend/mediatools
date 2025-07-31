import os
import cv2
import subprocess
import torch
import multiprocessing as mp
from functools import partial
import logging
import warnings
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock
import contextlib
import io
import shutil

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

temp_frames_dir = 'temp_frames'
enhanced_frames_dir = 'enhanced_frames'
gfpgan_model_path = 'Real-ESRGAN/weights/GFPGANv1.4.pth'
tqdm_lock = Lock()
BATCH_SIZE = 8  # Default batch size

# Define function for processing a chunk of frames
def process_chunk(chunk, temp_frames_dir, enhanced_frames_dir, model_path, use_mps, batch_size, denoise_enabled, denoise_before, scale, face_enhance_enabled, gfpgan_model_path):
    process_name = mp.current_process().name
    logger.info(f"{process_name} starting to process {len(chunk)} frames")

    device = torch.device('mps' if use_mps and torch.backends.mps.is_available() else 'cpu')
    logger.info(f"{process_name} using device: {device}")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=256,
            tile_pad=20,
            pre_pad=0,
            half=use_mps,
            device=device
        )

        face_enhancer = None
        if face_enhance_enabled:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path=gfpgan_model_path,
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler,
                device=device
            )

        with tqdm(total=(len(chunk) + batch_size - 1) // batch_size, desc=f"{process_name} Progress", position=mp.current_process()._identity[0]) as pbar:
            for i in range(0, len(chunk), batch_size):
                batch_indices = chunk[i:i + batch_size]
                for idx in batch_indices:
                    img_path = f'{temp_frames_dir}/frame_{idx:06d}.png'
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        logger.error(f"{process_name} failed to load frame {idx}")
                        continue
                    if denoise_enabled and denoise_before:
                        logger.info(f"Applying light denoising to frame {idx} before upscaling")
                        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Add denoising

                    with contextlib.redirect_stdout(io.StringIO()):
                        if face_enhance_enabled:
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                img_process = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            else:
                                img_process = img
                            logger.info(f"Starting face enhancement for frame {idx}")
                            cropped_faces, restored_faces, enhanced_img = face_enhancer.enhance(
                                img_process,
                                has_aligned=False,
                                only_center_face=False,
                                paste_back=True,
                                weight=0.8  # Reduce weight to preserve more original details
                            )

                            # Log return types for debugging
                            logger.info(f"cropped_faces type: {type(cropped_faces)}, length: {len(cropped_faces)}")
                            logger.info(f"restored_faces type: {type(restored_faces)}, length: {len(restored_faces)}")
                            logger.info(f"enhanced_img type: {type(enhanced_img)}, shape: {enhanced_img.shape if isinstance(enhanced_img, np.ndarray) else 'N/A'}")

                            # Convert enhanced_img to NumPy array if it's a tensor
                            if isinstance(enhanced_img, torch.Tensor):
                                enhanced_img = enhanced_img.cpu().numpy()
                                if enhanced_img.shape[0] in [3, 4]:
                                    enhanced_img = enhanced_img.transpose(1, 2, 0)
                                if enhanced_img.dtype != np.uint8:
                                    enhanced_img = (enhanced_img * 255).clip(0, 255).astype(np.uint8)

                            # Ensure enhanced_img is a NumPy array
                            if not isinstance(enhanced_img, np.ndarray):
                                logger.error(f"Invalid enhanced_img type: {type(enhanced_img)}")
                                continue

                            # Convert RGB to BGR for OpenCV
                            if len(enhanced_img.shape) == 3 and enhanced_img.shape[2] == 3:
                                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
                        else:
                            enhanced_img, _ = upsampler.enhance(img, outscale=scale)

                    if denoise_enabled and not denoise_before:
                        logger.info(f"Applying light denoising to frame {idx} after upscaling")
                        enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)  # Add denoising

                    cv2.imwrite(f'{enhanced_frames_dir}/frame_{idx:06d}.png', enhanced_img)
                with tqdm_lock:
                    pbar.update(1)
        logger.info(f"{process_name} finished processing {len(chunk)} frames")
    except Exception as e:
        logger.error(f"{process_name} encountered error: {str(e)}")
        raise

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    input_video = input("Enter the input video path: ").strip()
    if not os.path.exists(input_video):
        logger.error(f"Input video {input_video} does not exist.")
        exit(1)

    # Prompt for output file name
    default_output_base = f"NEW_{os.path.splitext(os.path.basename(input_video))[0]}"
    output_prompt = input(f"Enter the output video path (default: {default_output_base} in same directory): ").strip()
    if output_prompt:
        output_video = output_prompt
    else:
        # Compute default output path with iterator
        input_dir = os.path.dirname(input_video)
        input_base, input_ext = os.path.splitext(os.path.basename(input_video))
        output_base = f"NEW_{input_base}"
        i = 0
        while True:
            suffix = '' if i == 0 else f"_{i}"
            output_video = os.path.join(input_dir, f"{output_base}{suffix}{input_ext}")
            if not os.path.exists(output_video):
                break
            i += 1

    # Prompt for denoising
    denoise_input = input("Apply light denoising? (y/n, default n): ").strip().lower()
    denoise_enabled = denoise_input == 'y'

    # Prompt for denoising timing
    denoise_before = True
    if denoise_enabled:
        denoise_timing = input("Apply denoising before or after upscaling? (before/after, default before): ").strip().lower()
        denoise_before = denoise_timing != 'after'

    # Prompt for upscale choice, default to 4K (2x)
    upscale_choice = input("Upscale to 4K (2x) or 8K (4x)? (enter '8k' for 8K, default 4K): ").strip().lower()
    if '8k' in upscale_choice or '4x' in upscale_choice:
        scale = 4
        model_path = 'Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
    else:
        scale = 2
        model_path = 'Real-ESRGAN/weights/RealESRGAN_x2plus.pth'

    if not os.path.exists(model_path):
        logger.error(f"Model {model_path} does not exist.")
        exit(1)

    # Prompt for face enhancement
    face_enhance_input = input("Apply face enhancement? (y/n, default n): ").strip().lower()
    face_enhance_enabled = face_enhance_input == 'y'

    if face_enhance_enabled:
        if not os.path.exists(gfpgan_model_path):
            logger.error(f"GFPGAN model {gfpgan_model_path} does not exist.")
            exit(1)

    # Prompt for device, default to mps
    device_choice = input("Use MPS (if available) or CPU? (mps/cpu, default mps): ").strip().lower() or 'mps'
    use_mps = (device_choice == 'mps') and torch.backends.mps.is_available()

    if use_mps:
        num_processes = 1
    else:
        num_str = input("Enter number of processes (default 8): ").strip()
        num_processes = int(num_str) if num_str else 8

    # Prompt for cleanup
    cleanup_input = input("Clean up temporary frame directories after processing? (y/n, default y): ").strip().lower() or 'y'
    cleanup_enabled = cleanup_input == 'y'

    # Step 1: Extract frames
    logger.info("Starting frame extraction")
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(enhanced_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{temp_frames_dir}/frame_{frame_count:06d}.png', frame)
        frame_count += 1
    cap.release()
    logger.info(f"Extracted {frame_count} frames at {fps} FPS")

    chunk_size = (frame_count + num_processes - 1) // num_processes
    chunks = [range(i * chunk_size, min((i + 1) * chunk_size, frame_count)) for i in range(num_processes)]
    logger.info(f"Starting {num_processes} processes with chunk size {chunk_size}")

    try:
        with mp.Pool(num_processes) as p:
            p.map(partial(process_chunk, temp_frames_dir=temp_frames_dir,
                         enhanced_frames_dir=enhanced_frames_dir,
                         model_path=model_path,
                         use_mps=use_mps,
                         batch_size=BATCH_SIZE,
                         denoise_enabled=denoise_enabled,
                         denoise_before=denoise_before,
                         scale=scale,
                         face_enhance_enabled=face_enhance_enabled,
                         gfpgan_model_path=gfpgan_model_path), chunks)
    except Exception as e:
        logger.error(f"Multiprocessing failed: {str(e)}")
        raise

    logger.info("Reassembling video with FFmpeg")
    frame_pattern = f'{enhanced_frames_dir}/frame_%06d.png'
    result = subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps), '-i', frame_pattern,
        '-i', input_video, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
        output_video
    ])
    if result.returncode == 0:
        logger.info(f"Video saved to {output_video}")
        if cleanup_enabled:
            logger.info(f"Cleaning up temporary directories: {temp_frames_dir}, {enhanced_frames_dir}")
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)
            if os.path.exists(enhanced_frames_dir):
                shutil.rmtree(enhanced_frames_dir)
    else:
        logger.error(f"FFmpeg failed with return code {result.returncode}")
