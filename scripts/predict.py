"""Run ParsBERT inference offline from the command line.

Examples:
    python scripts/predict.py "کیفیت محصول خیلی خوب بود"
    Get-Content review.txt | python scripts/predict.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `python scripts/predict.py ...` work when called from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import ParsBERTPredictor, default_model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict Digikala review labels with ParsBERT")
    parser.add_argument("text", nargs="?", help="Review text; if omitted, read text from stdin")
    parser.add_argument(
        "--model-dir",
        default=str(default_model_dir()),
        help="Directory containing model.safetensors, tokenizer/, and model_metadata.json",
    )
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("provide review text as an argument or through stdin")

    try:
        predictor = ParsBERTPredictor.from_directory(args.model_dir)
        prediction = predictor.predict(text)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

