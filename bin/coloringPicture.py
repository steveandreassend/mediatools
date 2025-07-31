import cv2
import os
import numpy as np

def image_to_coloring_picture():
    # Prompt user for input image file
    input_image_path = input("Enter the path to the input image: ").strip()

    # Check if file exists
    if not os.path.isfile(input_image_path):
        print("Error: File not found.")
        return

    # Generate output filename (same name, different extension)
    base_name, _ = os.path.splitext(input_image_path)
    output_image_path = f"{base_name}_coloring.png"

    # Load the image in grayscale
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise and details
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)

    # Use Canny edge detection to get simpler outlines
    #edges = cv2.Canny(blurred_image, 50, 150)
    #edges = cv2.Canny(blurred_image, 35, 105)
    #edges = cv2.Canny(blurred_image, 35, 90)
    edges = cv2.Canny(blurred_image, 95, 90)

    # Invert colors to have black lines on white background
    inverted_image = cv2.bitwise_not(edges)

    # Save the result
    cv2.imwrite(output_image_path, inverted_image)
    print(f"Coloring-in picture saved at: {output_image_path}")

# Run the function
image_to_coloring_picture()
