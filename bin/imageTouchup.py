import cv2
import numpy as np
import os

def get_image_path():
    print("=== Gemini Symbol Remover (macOS) ===")
    print("Drag and drop your image file here, then press Enter:\n")

    path = input().strip().strip('"\'')

    if not os.path.exists(path):
        print("❌ File not found. Please try again.\n")
        return get_image_path()

    return path


def remove_gemini_symbol(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read the image. Try a different file.")
        return

    display_img = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    drawing = False

    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, mask, display_img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), 20, 255, -1)
            cv2.circle(display_img, (x, y), 20, (0, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(mask, (x, y), 20, 255, -1)

    print("\nInstructions:")
    print("• Click and drag over the Gemini symbol to mask it")
    print("• Press 'r' to reset")
    print("• Press Enter or Esc when done\n")

    cv2.namedWindow("Draw over Gemini Logo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw over Gemini Logo", 1200, 800)
    cv2.setMouseCallback("Draw over Gemini Logo", draw_mask)

    while True:
        cv2.imshow("Draw over Gemini Logo", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') or key == ord('R'):
            mask.fill(0)
            display_img = img.copy()
            print("Mask reset.")
        elif key in (13, 27):        # Enter or Esc
            break

    cv2.destroyAllWindows()

    if np.any(mask > 0):
        print("🧠 Removing symbol using inpainting...")
        result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Save output
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_no_gemini{ext}"

        cv2.imwrite(output_path, result)
        print(f"✅ Saved: {output_path}")

        # Show side-by-side comparison
        comparison = np.hstack([img, result])
        cv2.imshow("Before (Left)  —  After (Right)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No mask drawn.")

if __name__ == "__main__":
    image_path = get_image_path()
    remove_gemini_symbol(image_path)
