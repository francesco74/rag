import os
from pathlib import Path
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
# Official PyTorch Model ID (This link works 100%)
# https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
OUTPUT_DIR = Path("./model_cache/mmarco-mMiniLMv2-L12-H384-v1")

def build_model():
    print(f"🚀 Building FAST model from: {MODEL_ID}")
    
    # 1. Load PyTorch model and Export to ONNX
    print("Step 1/2: Downloading and Exporting to ONNX...")
    try:
        # This downloads the PyTorch weights and converts them to ONNX in one step
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            export=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Save the intermediate (float32) model
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return

    # 2. Quantize to INT8 (The secret to CPU speed)
    print("Step 2/2: Quantizing to INT8 (making it tiny and fast)...")
    try:
        # Create quantizer from the exported model
        quantizer = ORTQuantizer.from_pretrained(OUTPUT_DIR, file_name="model.onnx")
        
        # Define configuration (AVX512 is standard for modern servers)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        # Apply quantization
        quantizer.quantize(
            save_dir=OUTPUT_DIR,
            quantization_config=qconfig,
        )
        
        # Rename for compatibility with your worker
        # Your worker looks for 'model_quantized.onnx' or 'model.onnx'
        # The quantizer output is usually 'model_quantized.onnx'
        print(f"✓ Quantization complete.")
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        return

    print("\n✅ SUCCESS!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("Update your .env to point to this directory.")

if __name__ == "__main__":
    build_model()