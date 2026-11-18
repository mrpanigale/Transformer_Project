import html
import json
import re
import unicodedata
from pathlib import Path

import joblib
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک"})
URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
INVISIBLE_PATTERN = re.compile(r"[\u200b\ufeff\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def normalize_persian_text(text):
    """همان مراحل نرمال‌سازی سلول ۲۱ نوت‌بوک preprocess."""
    text = html.unescape(str(text)).translate(ARABIC_TO_PERSIAN)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = INVISIBLE_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


class TwoHeadClassifier(nn.Module):
    """معماری استفاده‌شده در آموزش: یک encoder و دو سر طبقه‌بندی."""

    def __init__(self, config, targets):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.heads = nn.ModuleDict({
            target: nn.Linear(config.hidden_size, len(classes))
            for target, classes in targets.items()
        })

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return {target: head(pooled) for target, head in self.heads.items()}


class Predictor:
    def __init__(self):
        self.tfidf = {}
        self.transformers = {}

    def load_models(self):
        """مدل‌ها را یک بار هنگام شروع سرویس بارگذاری می‌کند."""
        tfidf_dir = MODELS_DIR / "tf_idf"
        for target in ("recommendation_status", "rating_band"):
            self.tfidf[target] = joblib.load(tfidf_dir / f"{target}.joblib")

        # هر دو مدل از یک checkpoint و معماری پایه استفاده می‌کنند؛ تنظیمات
        # encoder به‌صورت محلی همراه مدل نیمه‌نظارتی ذخیره شده است.
        config_dir = MODELS_DIR / "semi_supervised" / "artifacts" / "encoder_config"
        for name, directory in (
            ("transformer", "parsbert_finetuning"),
            ("semi_supervised", "semi_supervised"),
        ):
            artifact_dir = MODELS_DIR / directory / "artifacts"
            with (artifact_dir / "model_metadata.json").open(encoding="utf-8") as file:
                metadata = json.load(file)
            tokenizer = AutoTokenizer.from_pretrained(artifact_dir / metadata["tokenizer_dir"], local_files_only=True)
            config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
            model = TwoHeadClassifier(config, metadata["targets"])
            model.load_state_dict(load_file(str(artifact_dir / metadata["weight_file"]), device="cpu"))
            model.eval()
            self.transformers[name] = {"model": model, "tokenizer": tokenizer, "metadata": metadata}

    def predict(self, text, model_name):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty.")
        if model_name not in ("tfidf", "transformer", "semi_supervised"):
            raise ValueError("Unknown model.")
        normalized_text = normalize_persian_text(text)
        if not normalized_text:
            raise ValueError("Text is empty after normalization.")

        results = {}
        if model_name == "tfidf":
            for target, model in self.tfidf.items():
                probabilities = model.predict_proba([normalized_text])[0]
                results[target] = make_result(model.classes_, probabilities)
        else:
            bundle = self.transformers[model_name]
            inputs = bundle["tokenizer"](
                normalized_text, truncation=True,
                max_length=bundle["metadata"]["max_length"], return_tensors="pt"
            )
            with torch.no_grad():
                logits = bundle["model"](inputs["input_ids"], inputs["attention_mask"])
            for target, values in logits.items():
                probabilities = torch.softmax(values[0], dim=-1).tolist()
                results[target] = make_result(bundle["metadata"]["targets"][target], probabilities)

        return {"model": model_name, "normalized_text": normalized_text, **results}


def make_result(classes, probabilities):
    scores = {str(label): float(probability) for label, probability in zip(classes, probabilities)}
    return {"label": max(scores, key=scores.get), "probabilities": scores}
