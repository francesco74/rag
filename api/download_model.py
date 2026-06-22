from huggingface_hub import snapshot_download
import os, shutil

def setup_model():
    # 1. Use the official or community ONNX repo
    #repo_id = "onnx-community/bge-reranker-v2-m3-ONNX"
    #local_dir = "./model_cache/bge-reranker-v2-m3-ONNX-int8"
    repo_id = "Xenova/jina-reranker-v2-base-multilingual"
    local_dir = "./model_cache/jina-reranker-v2-base-multilingual"

    print(f"Downloading {repo_id} to {local_dir}...")

    # 2. Download Model AND Tokenizer files
    # We need *.json and *.txt for the tokenizer (vocab, special tokens, etc.)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*.onnx", "*.json", "*.txt", "*.model"],
        local_dir_use_symlinks=False  # Important for Docker: get real files, not symlinks
    )

    # 3. Locate the Quantized Model
    # In 'onnx-community' repos, the file is usually at the ROOT.
    # We check multiple common locations to be safe.
    potential_paths = [
        os.path.join(local_dir, "model_quantized.onnx"),
        os.path.join(local_dir, "model.onnx"),
        os.path.join(local_dir, "onnx", "model_quantized.onnx"),
    ]

    found_path = None
    for p in potential_paths:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        print(f"Error: Quantized ONNX file not found in {local_dir}")
        # List files to debug what actually downloaded
        print("Files downloaded:", os.listdir(local_dir))
        exit(1)

    # 4. Standardize the Filename
    # If you are using your custom ONNX class, 'model.onnx' is a standard name.
    # If using FlashRank, it specifically looks for 'flashrank-quantized.onnx'.
    final_name = "model_quantized.onnx" 
    dst = os.path.join(local_dir, final_name)

    if found_path != dst:
        print(f"Renaming {found_path} -> {dst}")
        shutil.move(found_path, dst)

    print("✅ Model setup complete.")
    print(f"Path: {dst}")

if __name__ == "__main__":
    setup_model()