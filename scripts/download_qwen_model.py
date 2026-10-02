"""Download Qwen2.5-Omni weights into the configured Hugging Face cache."""

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


if __name__ == "__main__":
    path = snapshot_download(MODEL_ID)
    print(f"Download complete: {path}")
