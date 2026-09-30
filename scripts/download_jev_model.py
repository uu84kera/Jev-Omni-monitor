"""Download Jev-Omni weights into the configured Hugging Face cache."""

from huggingface_hub import snapshot_download


MODEL_ID = "akhilaaa3/Jev-Omni"
ALLOW_PATTERNS = [
    "chat_template.jinja",
    "config.json",
    "decision_config.json",
    "generation_config.json",
    "head.pt",
    "jev_omni.py",
    "model*.safetensors*",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


if __name__ == "__main__":
    path = snapshot_download(MODEL_ID, allow_patterns=ALLOW_PATTERNS)
    print(path)
