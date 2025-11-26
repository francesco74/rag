"""
Handwritten OCR using Microsoft TrOCR (Hugging Face)
----------------------------------------------------
Performs high-quality handwriting OCR on scanned or photographed documents.
Includes preprocessing for improved accuracy.

Requirements:
    pip install torch torchvision transformers pillow opencv-python numpy

Usage:
    python handwriting_ocr.py
"""

import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import os


# -------------------------------------------------------------
# STEP 1: Preprocess image (grayscale, denoise, threshold, resize)
# -------------------------------------------------------------
def preprocess_image(input_path, output_path="processed.jpg"):
    print("[INFO] Preprocessing image...")

    # Load image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Remove noise using Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Adaptive threshold for better contrast
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    # Optional: resize image for better OCR accuracy
    scale_factor = 2.0
    img = cv2.resize(
        img,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # Save processed image
    cv2.imwrite(output_path, img)
    print(f"[INFO] Processed image saved as: {output_path}")
    return output_path


# -------------------------------------------------------------
# STEP 2: Load model and run TrOCR
# -------------------------------------------------------------
def run_trocr_ocr(image_path):
    print("[INFO] Loading TrOCR model (this may take a bit on first run)...")

    # Load pre-trained model and processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Load preprocessed image
    image = Image.open(image_path).convert("RGB")

    # Convert to tensor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Run model
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    # Decode output
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# -------------------------------------------------------------
# STEP 3: Save recognized text
# -------------------------------------------------------------
def save_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Recognized text saved to: {output_path}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # Input image path
    INPUT_IMAGE = "_DSC2339-2.jpg"  # change if needed

    # Preprocess
    processed_image = preprocess_image(INPUT_IMAGE)

    # OCR
    recognized_text = run_trocr_ocr(processed_image)

    # Output
    OUTPUT_TEXT = os.path.splitext(INPUT_IMAGE)[0] + "_recognized.txt"
    save_text(recognized_text, OUTPUT_TEXT)

    print("\n===== RECOGNIZED TEXT =====\n")
    print(recognized_text)
    print("\n===========================\n")

