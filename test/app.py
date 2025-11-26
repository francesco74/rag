import easyocr
import time

# --- IMPORTANT ---
# This script requires easyocr and its dependencies (like torch).
# Install with:
# pip install easyocr torch
#
# The first time you run this, it will download the models for the
# specified language (e.g., 'en' for English).

def ocr_with_easyocr(image_path):
    """
    Performs OCR on an image using EasyOCR, which is often
    better for unstructured or handwritten text.
    """
    print("\n--- Running OCR with EasyOCR ---")
    
    try:
        # Initialize the EasyOCR reader. 
        # This will download the model if not already cached.
        print("Initializing EasyOCR reader (this may take a moment)...")
        # We specify English ['en'] as the language
        reader = easyocr.Reader(['en']) 
        print("Reader initialized.")
        
        # Read the text from the image
        # 'detail=0' returns only the text as a list of strings
        # 'paragraph=True' tries to stitch text blocks together
        print(f"Reading text from '{image_path}'...")
        start_time = time.time()
        results = reader.readtext(image_path, detail=0, paragraph=True)
        end_time = time.time()
        
        print(f"OCR completed in {end_time - start_time:.2f} seconds.")

        print("\n--- Detected Text (EasyOCR) ---")
        if results:
            for line in results:
                print(line)
        else:
            print("[No text detected]")
        print("---------------------------------")

    except ImportError:
        print("\nError: 'easyocr' or 'torch' library not found.")
        print("Please install with: pip install easyocr torch")
    except RuntimeError as e:
        print(f"\nA runtime error occurred (e.g., CUDA issue or model download failed): {e}")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred during EasyOCR: {e}")

if __name__ == "__main__":
    # Replace with the path to your image
    image_file = "_DSC2339-2.jpg"
    ocr_with_easyocr(image_file)

