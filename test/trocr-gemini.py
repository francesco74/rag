import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2  # We'll use OpenCV for line detection
import numpy as np
import time
import traceback
import os # Import os to handle file paths
import math # For skew correction

# --- IMPORTANT ---
# This script now requires 'opencv-python' and 'numpy'.
# Install with:
# pip install transformers torch pillow opencv-python-headless numpy
#
# NOTE: The first time you run this, it will download the TrOCR model
# (e.g., 'microsoft/trocr-large-handwritten'), which is several hundred MB.

def load_model():
    """Loads the TrOCR processor and model."""
    # --- UPGRADE: Using the 'large' model for better accuracy ---
    model_name = "microsoft/trocr-large-handwritten"
    print(f"Loading processor for '{model_name}'...")
    print("Note: Using the 'large' model. This is more accurate but will be a")
    print("larger download (~2GB) and slower to run.")
    processor = TrOCRProcessor.from_pretrained(model_name)
    print(f"Loading model '{model_name}'...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    print("TrOCR model loaded successfully.")
    return processor, model

def correct_skew(image_cv):
    """
    Detects and corrects skew (rotation) in the document.
    Returns a deskewed OpenCV image.
    """
    print("Correcting document skew...")
    # Load image, grayscale, and binarize
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Invert the image (white text on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Find main text block ---
    # Use findContours to find the largest "blob" of text
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Skew correction: No contours found. Returning original image.")
        return image_cv
        
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding rectangle
    # This rectangle will be rotated to fit the text block
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    
    # --- Normalize the angle ---
    # The angle from minAreaRect can be in [-90, 0).
    # We want to get the angle relative to the horizontal axis.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    if abs(angle) < 1.0:
        print(f"Skew angle {angle:.2f} is minimal. Skipping rotation.")
        return image_cv
        
    print(f"Detected skew angle: {angle:.2f} degrees. Rotating...")
    
    # Get rotation matrix
    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    # Use a white background to fill in new areas
    deskewed = cv2.warpAffine(image_cv, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
                              
    return deskewed

def get_text_lines(image_path):
    """
    Uses OpenCV to detect lines of text in the image and returns
    a list of cropped (Pillow) images, one for each line.
    """
    print(f"Loading image with OpenCV for line detection...")
    # Load image with OpenCV
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Could not load image at {image_path} with OpenCV.")
    
    # --- NEW: Correct Skew ---
    image_cv = correct_skew(image_cv)
    # --------------------------
        
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image (black text on white background)
    # Using Otsu's method to automatically find the threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Now we have white text on a black background
    
    # --- Morphological Operations to Detect Lines ---
    # We want to connect text horizontally to form lines
    # but not vertically.
    
    # Create a long horizontal kernel.
    # *** You may need to tune 'kernel_width' and 'iterations'
    # *** based on your document's font size and line spacing.
    kernel_height = 5
    kernel_width = 40 # This width connects words in a line
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    
    # Dilate the image to connect nearby text components
    dilated = cv2.dilate(binary, kernel, iterations=3) # 3 iterations to connect words

    # --- Find Contours ---
    # Find contours of the dilated text blocks (lines)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} potential text lines.")
    
    # --- Filter and Sort Contours ---
    line_bboxes = []
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small (noise) or very large (full page) contours
        # These values may need tuning for your specific document
        min_width = 50
        min_height = 10
        if w > min_width and h > min_height:
            line_bboxes.append((x, y, w, h))
            
    # Sort the bounding boxes by their top-left y-coordinate (top-to-bottom)
    line_bboxes.sort(key=lambda bbox: bbox[1])
    
    print(f"Filtered to {len(line_bboxes)} lines, sorted top-to-bottom.")

    # --- Crop Images ---
    # --- NEW: Setup debug output directory ---
    debug_dir = "line_debug_output"
    if os.path.exists(debug_dir):
        # Clear old debug files
        for f in os.listdir(debug_dir):
            os.remove(os.path.join(debug_dir, f))
    else:
        os.makedirs(debug_dir, exist_ok=True)
    # ----------------------------------------

    line_images = []
    # Load the original image (deskewed) with Pillow for high-quality cropping
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    padding = 10 # Add 10px padding around each line
    
    for i, (x, y, w, h) in enumerate(line_bboxes):
        # Get coordinates with padding, ensuring they are within image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_pil.width, x + w + padding)
        y2 = min(image_pil.height, y + h + padding)
        
        # Crop the image from the *original* PIL image
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        line_images.append(cropped_image)
        
        # --- NEW: Save debug image ---
        try:
            debug_filename = os.path.join(debug_dir, f"line_{i:03d}.png")
            cropped_image.save(debug_filename)
        except Exception as e:
            print(f"Warning: Could not save debug image {debug_filename}. {e}")
        # -----------------------------
        
    print(f"Saved {len(line_images)} cropped line images to '{debug_dir}' for debugging.")
    return line_images

def ocr_with_trocr_production(image_path):
    """
    Performs production-ready OCR by first detecting lines
    and then running TrOCR on each line individually.
    """
    print("\n--- Running Production-Ready TrOCR ---")
    
    try:
        # 1. Load the TrOCR model (once)
        processor, model = load_model()
        
        # 2. Get all text line images
        line_images = get_text_lines(image_path)
        
        if not line_images:
            print("No text lines were detected. Exiting.")
            return

        print(f"\n--- Recognizing text from {len(line_images)} lines ---")
        
        full_text = []
        start_time = time.time()
        
        # 3. Loop and recognize each line
        for i, line_img in enumerate(line_images):
            print(f"Processing line {i+1}/{len(line_images)}...")
            
            # 'pixel_values' is the image tensor ready for the model
            pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
            
            # 'generated_ids' are the token IDs of the predicted text
            generated_ids = model.generate(pixel_values)
            
            # 'generated_text' is the final, human-readable string
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            full_text.append(generated_text)
            print(f"  > {generated_text}")

        end_time = time.time()
        
        print("\n--- OCR Complete ---")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        if line_images:
            print(f"Average time per line: {(end_time - start_time) / len(line_images):.2f} seconds.")
        
        # --- NEW: Save text to file ---
        # Create an output filename, e.g., "output.txt"
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_filename}_output.txt"
        
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(full_text))
            print(f"\nSuccessfully saved text to {output_filename}")
        except IOError as e:
            print(f"\nError: Could not write to file {output_filename}. {e}")
        # --- End of new section ---

        print("\n--- Full Detected Text (in order) ---")
        for line in full_text:
            print(line)
        print("---------------------------------------")

    except ImportError:
        print("\nError: Required libraries not found.")
        print("Please install with: pip install transformers torch pillow opencv-python-headless numpy")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Path to your handwritten image
    image_file = "_DSC2339-2.jpg"
    ocr_with_trocr_production(image_file)


