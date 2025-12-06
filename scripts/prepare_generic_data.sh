mkdir -p scripts data

cat > scripts/prepare_generic_wikitext.py << 'EOF'
from datasets import load_dataset
from pathlib import Path

OUT_PATH = Path("../data/generic_calibration.txt")
OUT_PATH.parent.mkdir(exist_ok=True)

# Load the WikiText-2 raw dataset from Hugging Face
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

with OUT_PATH.open("w", encoding="utf-8") as f:
    for ex in ds:
        text = ex["text"].strip()
        if text:
            f.write(text + "\n")
EOF
