import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def get_unique_filename(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def image_to_coloring_picture():
    # Prompt user for input image file
    input_image_path = input("Enter the path to the input image: ").strip()

    # Check if file exists
    if not os.path.isfile(input_image_path):
        print("Error: File not found.")
        return

    # Generate output filename (same name, different extension)
    base_name, _ = os.path.splitext(input_image_path)
    proposed_image_path = f"{base_name}_coloring.png"
    output_image_path = get_unique_filename(proposed_image_path)

    # Load the image in grayscale
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise and details
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)

    # Use Canny edge detection to get simpler outlines
    edges = cv2.Canny(blurred_image, 95, 90)

    # Invert colors to have black lines on white background
    inverted_image = cv2.bitwise_not(edges)

    # Save the result
    cv2.imwrite(output_image_path, inverted_image)
    print(f"Coloring-in picture saved at: {output_image_path}")

    # Prompt for PDF generation
    pdf_prompt = input("Generate PDF? (yes/no, default no): ").strip().lower()
    if pdf_prompt != 'yes':
        return

    # Generate proposed PDF path
    proposed_pdf_path = f"{base_name}_coloring.pdf"
    output_pdf_path = get_unique_filename(proposed_pdf_path)

    # Load original image in color (RGB for matplotlib)
    original_bgr = cv2.imread(input_image_path)
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # Determine orientation
    h, w = original.shape[:2]
    image_aspect = w / h
    is_landscape = w > h
    if is_landscape:
        figsize = (11.6929, 8.2677)
    else:
        figsize = (8.2677, 11.6929)

    # Calculate figure aspect (height / width)
    fig_aspect = figsize[1] / figsize[0]

    # Calculate margins
    margin_mm = 2
    margin_in = margin_mm / 25.4
    margin_frac_h = margin_in / figsize[0]
    margin_frac_v = margin_in / figsize[1]

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Calculate fractions for small original (approx 1/10th area)
    area_frac = 0.1
    required_frac_ratio = image_aspect * fig_aspect
    small_h_frac = np.sqrt(area_frac / required_frac_ratio)
    small_w_frac = required_frac_ratio * small_h_frac

    # Cap if exceeds page
    scale = min((1 - 2 * margin_frac_h) / small_w_frac, (1 - 2 * margin_frac_v) / small_h_frac)
    if scale < 1:
        small_w_frac *= scale
        small_h_frac *= scale

    # Position small original at top left
    left_small = margin_frac_h
    bottom_small = 1 - small_h_frac - margin_frac_v
    ax_small = fig.add_axes([left_small, bottom_small, small_w_frac, small_h_frac])
    ax_small.imshow(original)
    ax_small.axis('off')

    # Calculate available space for coloring image
    margin_between = margin_frac_v
    max_h_frac_color = bottom_small - margin_between - margin_frac_v
    max_w_frac_color = 1 - 2 * margin_frac_h

    # Fit coloring image to available space
    box_frac_ratio = max_w_frac_color / max_h_frac_color
    if required_frac_ratio > box_frac_ratio:
        coloring_w_frac = max_w_frac_color
        coloring_h_frac = coloring_w_frac / required_frac_ratio
    else:
        coloring_h_frac = max_h_frac_color
        coloring_w_frac = coloring_h_frac * required_frac_ratio

    # Position coloring below original, centered horizontally
    coloring_top = bottom_small - margin_between
    coloring_bottom = coloring_top - coloring_h_frac
    left_color = (1 - coloring_w_frac) / 2
    ax_color = fig.add_axes([left_color, coloring_bottom, coloring_w_frac, coloring_h_frac])
    ax_color.imshow(inverted_image, cmap='gray')
    ax_color.axis('off')

    # Save PDF
    fig.savefig(output_pdf_path)
    print(f"PDF saved at: {output_pdf_path}")

# Run the function
image_to_coloring_picture()
