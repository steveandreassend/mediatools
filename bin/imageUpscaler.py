import os
import cv2
import torch
import logging
import warnings
import multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(processName)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

gfpgan_model_path = 'Real-ESRGAN/weights/GFPGANv1.4.pth'

def process_image(input_image, output_image, model_path, gfpgan_model_path, scale, denoise_enabled):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from gfpgan import GFPGANer

        # Initialize Real-ESRGAN model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=256,
            tile_pad=20,
            pre_pad=0,
            half=True if device.type == 'mps' else False,
            device=device
        )

        # Initialize GFPGAN for face enhancement
        face_enhancer = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device
        )

        logger.info(f"Loading image: {input_image}")
        img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to load image {input_image}")
            return

        # Apply denoising if enabled
        if denoise_enabled:
            logger.info("Applying light denoising to input image")
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 15)

        # Convert BGR to RGB for GFPGAN
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        logger.info("Starting image upscaling with face enhancement")
        cropped_faces, restored_faces, enhanced_img = face_enhancer.enhance(
            img,
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
            return

        # Convert RGB to BGR for OpenCV
        if enhanced_img.shape[2] == 3:
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_image, enhanced_img)
        logger.info(f"Enhanced image saved to {output_image}")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    input_image = input("Enter the input image path: ").strip()
    if not os.path.exists(input_image):
        logger.error(f"Input image {input_image} does not exist.")
        exit(1)

    # Prompt for denoising
    denoise_input = input("Apply light denoising? (y/n, default n): ").strip().lower()
    denoise_enabled = denoise_input == 'y'

    # Prompt for upscale choice
    upscale_choice = input("Upscale to 8K (4x) or 4K (2x)? (enter '4k' for 4K, default 8K): ").strip().lower()
    if '4k' in upscale_choice or '2x' in upscale_choice:
        scale = 2
        model_path = 'Real-ESRGAN/weights/RealESRGAN_x2plus.pth'
    else:
        scale = 4
        model_path = 'Real-ESRGAN/weights/RealESRGAN_x4plus.pth'

    if not os.path.exists(model_path):
        logger.error(f"Model {model_path} does not exist.")
        exit(1)
    if not os.path.exists(gfpgan_model_path):
        logger.error(f"GFPGAN model {gfpgan_model_path} does not exist.")
        exit(1)

    # Compute default output path with iterator
    input_dir = os.path.dirname(input_image)
    input_base, input_ext = os.path.splitext(os.path.basename(input_image))
    output_base = f"NEW_{input_base}"
    i = 0
    while True:
        suffix = '' if i == 0 else f"_{i}"
        output_image = os.path.join(input_dir, f"{output_base}{suffix}{input_ext}")
        if not os.path.exists(output_image):
            break
        i += 1

    with mp.Pool(1) as p:
        p.apply(process_image, args=(input_image, output_image, model_path, gfpgan_model_path, scale, denoise_enabled))
