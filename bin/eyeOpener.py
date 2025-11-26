import logging
import sys
import warnings

import cv2
import mediapipe as mp
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

# --- 1. SETUP LOGGING AND SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (Defaults) ---
MODEL_ID = "runwayml/stable-diffusion-inpainting"
TARGET_AI_SIZE = (512, 512)

# Base prompts are defined here as DEFAULTS
DEFAULT_POSITIVE = "open eyes, visible iris, visible pupil, detailed staring at camera, high quality, photorealistic, sharp focus"
DEFAULT_NEGATIVE = "closed eyes, sleeping, blinking, eyelids, sunglasses, cartoon, blurry, low quality, deformed, asymmetry, poorly lit"
DEFAULT_STRENGTH = 0.80  # Set blending strength to 0.80 as requested

# --- HELPER FUNCTIONS ---


def get_face_mesh_mask(crop_img, padding=10):
    """
    Detects eyes in the original crop, then returns the mask scaled to TARGET_AI_SIZE.
    """
    mp_face_mesh = mp.solutions.face_mesh
    h, w, _ = crop_img.shape
    mask_orig = np.zeros((h, w), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            logger.error(
                "❌ MediaPipe failed to find eyes in crop. Please run again and crop **TIGHTER** around the face."
            )
            return None

        # Precise Eye Indices
        LEFT_EYE = [
            362,
            382,
            381,
            380,
            374,
            373,
            390,
            249,
            263,
            466,
            388,
            387,
            386,
            385,
            384,
            398,
        ]
        RIGHT_EYE = [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            133,
            173,
            157,
            158,
            159,
            160,
            161,
            246,
        ]

        for face_landmarks in results.multi_face_landmarks:
            for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                points = []
                for idx in eye_indices:
                    lm = face_landmarks.landmark[idx]
                    points.append((int(lm.x * w), int(lm.y * h)))

                points = np.array(points)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask_orig, hull, 255)

    # Apply dilation and blur to the original mask
    kernel = np.ones((padding, padding), np.uint8)
    mask_orig = cv2.dilate(mask_orig, kernel, iterations=2)
    mask_blur = cv2.GaussianBlur(mask_orig, (15, 15), 8)

    # Scale the mask to the AI's preferred size (512x512)
    mask_scaled = cv2.resize(mask_blur, TARGET_AI_SIZE, interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(mask_scaled)


def manual_crop(img_orig):
    logger.info("waiting for manual selection...")
    h, w = img_orig.shape[:2]

    target_w = 1400
    scale = target_w / w if w > target_w else 1.0
    img_display = cv2.resize(img_orig, None, fx=scale, fy=scale)

    print("\n" + "=" * 50)
    print(" 1. A window will open showing your image.")
    print(" 2. Use your mouse to DRAW A BOX around the face.")
    print(" 3. Press SPACE or ENTER to confirm selection.")
    print("=" * 50 + "\n")

    roi = cv2.selectROI(
        "SELECT FACE (Space to Confirm)", img_display, showCrosshair=True
    )
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    x, y, w_box, h_box = roi

    if w_box == 0 or h_box == 0:
        logger.warning("Selection cancelled.")
        return None, None, None

    x_real = int(x / scale)
    y_real = int(y / scale)
    w_real = int(w_box / scale)
    h_real = int(h_box / scale)

    # Add context padding
    pad = int(w_real * 0.3)
    y1 = max(0, y_real - pad)
    y2 = min(h, y_real + h_real + pad)
    x1 = max(0, x_real - pad)
    x2 = min(w, x_real + w_real + pad)

    crop_img = img_orig[y1:y2, x1:x2]
    return crop_img, (x1, y1, x2, y2), scale


# --- MAIN EXECUTION ---


def main():
    print("\n=== AI Eye Opener (Fully Dynamic Prompt Version) ===\n")

    path = input("Input image path: ").strip().strip("\"'")
    if not path:
        return

    img_orig = cv2.imread(path)
    if img_orig is None:
        logger.error("Could not load image.")
        return

    # 1. Manual Selection
    crop_bgr, coords, _ = manual_crop(img_orig)
    if crop_bgr is None:
        return

    x1, y1, x2, y2 = coords
    crop_h, crop_w = crop_bgr.shape[:2]

    # --- FULLY DYNAMIC PROMPT INPUTS ---

    # 2. Get Core Positive Prompt (e.g., quality, style)
    print("\n--- Core Positive Prompt ---")
    core_prompt = (
        input(
            f"Enter core positive description (Default: '{DEFAULT_POSITIVE}'): "
        ).strip()
        or DEFAULT_POSITIVE
    )

    # 3. Get Negative Prompt
    print("\n--- Negative Prompt ---")
    negative_prompt = (
        input(f"Enter negative description (Default: '{DEFAULT_NEGATIVE}'): ").strip()
        or DEFAULT_NEGATIVE
    )

    # 4. Get Gender Modifier
    print("\n--- Gender Modifier ---")
    gender = input("Enter gender (e.g., female, male): ").strip().lower()
    gender_modifier = f" {gender}" if gender else ""

    # 5. Get Ethnicity Modifier
    print("\n--- Ethnicity Modifier ---")
    ethnicity = (
        input("Enter ethnicity (e.g., Asian, Caucasian, Black, Hispanic): ")
        .strip()
        .lower()
    )
    ethnicity_modifier = f" {ethnicity}" if ethnicity else ""

    # 6. Get Eye Color (Modifier)
    print("\n--- Eye Color Modifier ---")
    eye_color = (
        input("Enter natural eye color (e.g., brown, black, dark hazel): ")
        .strip()
        .lower()
    )
    color_modifier = f" {eye_color} iris, {eye_color} eyes" if eye_color else ""

    # 7. Get Environmental Context (Modifier)
    print("\n--- Environmental Context Modifier ---")
    environment_context = (
        input(
            "Enter lighting context (e.g., 'harsh light, strong shadows' or 'soft studio lighting'): "
        )
        .strip()
        .lower()
    )
    context_modifier = f", {environment_context}" if environment_context else ""

    # 8. Get Final Refinement (Modifier)
    print("\n--- Final Refinement Modifier ---")
    refinement_prompt = (
        input(
            "Enter any additional quality words (optional, e.g., 'award-winning photo, hyperdetailed, 8k'): "
        )
        .strip()
        .lower()
    )
    refinement_modifier = f", {refinement_prompt}" if refinement_prompt else ""

    # Final combined prompt construction
    final_prompt = f"{core_prompt}{gender_modifier}{ethnicity_modifier}{color_modifier}{context_modifier}{refinement_modifier}"

    logger.info(f"Final Positive Prompt: {final_prompt}")
    logger.info(f"Final Negative Prompt: {negative_prompt}")

    if crop_h < 32 or crop_w < 32:
        logger.error("Selection too small! Please select a larger area.")
        return

    logger.info(f"✅ Selected Area: {crop_w}x{crop_h} pixels. Upscaling for AI...")

    # Upscale the input image to the AI's preferred size (512x512)
    crop_bgr_upscaled = cv2.resize(
        crop_bgr, TARGET_AI_SIZE, interpolation=cv2.INTER_LANCZOS4
    )
    init_image_512 = Image.fromarray(cv2.cvtColor(crop_bgr_upscaled, cv2.COLOR_BGR2RGB))

    # 9. Setup AI
    logger.info("🚀 Loading AI Brain...")
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16"
        ).to("mps")
        pipe.enable_attention_slicing()

        # CRITICAL FIX: Disable NSFW Filter
        def dummy_safety_checker(images, clip_input):
            return images, [False]

        pipe.safety_checker = dummy_safety_checker

    except Exception as e:
        logger.error(f"AI Load Failed: {e}")
        return

    # 10. Masking
    mask_pil_512 = get_face_mesh_mask(crop_bgr)
    if mask_pil_512 is None:
        return

    # 11. Generate (Guaranteed 512x512 operation)
    logger.info("✨ Generating open eyes (at 512x512 resolution)...")

    generator = torch.Generator("mps").manual_seed(12345)

    output_crop_512 = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=init_image_512,
        mask_image=mask_pil_512,
        height=TARGET_AI_SIZE[1],
        width=TARGET_AI_SIZE[0],
        guidance_scale=9.0,
        num_inference_steps=40,
        strength=DEFAULT_STRENGTH,  # Using 0.80 strength as requested
        generator=generator,
        output_type="pil",
    ).images[0]

    # 12. Paste Back
    logger.info("🎨 Downscaling and stitching face back into group photo...")

    output_bgr_512 = cv2.cvtColor(np.array(output_crop_512), cv2.COLOR_RGB2BGR)

    # Downscale the fixed 512x512 image back to the original crop size
    output_bgr_resized = cv2.resize(
        output_bgr_512, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4
    )

    final_img = img_orig.copy()
    final_img[y1:y2, x1:x2] = output_bgr_resized

    out_path = path.replace(".", "_FIXED.")
    cv2.imwrite(out_path, final_img)

    logger.info(f"✅ DONE! Saved to: {out_path}")


if __name__ == "__main__":
    main()
