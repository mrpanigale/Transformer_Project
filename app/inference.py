"""Load and run the fine-tuned two-head ParsBERT model on CPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from app.normalization import normalize_persian_text


class ArtifactError(RuntimeError):
    """Base class for unavailable or invalid model artifacts."""


class ArtifactsMissingError(ArtifactError):
    """Raised when the expected model files are not present."""


class ArtifactLoadError(ArtifactError):
    """Raised when model artifacts cannot be loaded consistently."""


class TextTooLongError(ValueError):
    """Raised when normalized input exceeds the model's supported length."""


class TwoHeadParsBERT(nn.Module):
    """Architecture matching the fine-tuning notebook's TwoHeadParsBERT."""

    def __init__(
        self,
        encoder: nn.Module,
        targets: dict[str, list[str]],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, len(labels)) for name, labels in targets.items()}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = self.dropout(output.last_hidden_state[:, 0])
        return {name: head(pooled) for name, head in self.heads.items()}


class ParsBERTPredictor:
    """CPU predictor loaded once and reused by the CLI and API."""

    REQUIRED_TARGETS = {"recommendation_status", "rating_band"}

    def __init__(
        self,
        model: TwoHeadParsBERT,
        tokenizer: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.targets: dict[str, list[str]] = metadata["targets"]
        self.max_length = int(metadata["max_length"])
        self.model_name = str(metadata.get("model_name", "ParsBERT"))
        self.model_version = str(
            metadata.get("model_version")
            or metadata.get("training_config", {}).get("run_id")
            or "fine-tuned-parsbert"
        )

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "ParsBERTPredictor":
        directory = Path(model_dir).expanduser().resolve()
        metadata_path = directory / "model_metadata.json"
        metadata: dict[str, Any]

        if not directory.is_dir():
            raise ArtifactsMissingError(f"Model directory does not exist: {directory}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ArtifactsMissingError("model_metadata.json is missing") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactLoadError("model_metadata.json is unreadable or invalid") from exc

        weight_path = directory / str(metadata.get("weight_file", "model.safetensors"))
        tokenizer_dir = directory / str(metadata.get("tokenizer_dir", "tokenizer"))
        missing = [str(path.name) for path in (weight_path,) if not path.is_file()]
        if not tokenizer_dir.is_dir():
            missing.append(tokenizer_dir.name)
        if missing:
            raise ArtifactsMissingError("Missing model files: " + ", ".join(missing))

        targets = metadata.get("targets")
        if not isinstance(targets, dict) or not cls.REQUIRED_TARGETS.issubset(targets):
            raise ArtifactLoadError("Metadata must define both classifier heads and their label order")
        if any(not isinstance(labels, list) or not labels for labels in targets.values()):
            raise ArtifactLoadError("Each target must contain an ordered, non-empty label list")

        try:
            max_length = int(metadata["max_length"])
            if max_length < 2:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactLoadError("Metadata has an invalid max_length") from exc

        base_checkpoint = metadata.get("base_checkpoint")
        if not isinstance(base_checkpoint, str) or not base_checkpoint.strip():
            raise ArtifactLoadError("Metadata has no base_checkpoint")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                use_fast=True,
                local_files_only=True,
            )

            # The current training export contains the full fine-tuned state dict,
            # but does not yet include the encoder config. Prefer a local config
            # when available; otherwise fetch only the small config from the base
            # checkpoint and load all weights from model.safetensors below.
            config_dir = directory / str(metadata.get("encoder_config_dir", "encoder_config"))
            config_source = config_dir if (config_dir / "config.json").is_file() else base_checkpoint
            config = AutoConfig.from_pretrained(config_source)
            encoder = AutoModel.from_config(config)

            model = TwoHeadParsBERT(
                encoder=encoder,
                targets=targets,
                dropout=float(metadata.get("classifier_dropout", 0.1)),
            )
            state_dict = load_file(str(weight_path), device="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.to(torch.device("cpu"))
            model.eval()
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(
                "Could not load the tokenizer, encoder config, or model weights"
            ) from exc

        return cls(model=model, tokenizer=tokenizer, metadata=metadata)

    def tokenize(self, text: str) -> dict[str, Any]:
        normalized = normalize_persian_text(text)
        if not normalized:
            raise ValueError("text must not be empty after normalization")

        encoded = self.tokenizer(
            normalized,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        token_count = len(encoded["input_ids"])
        if token_count > self.max_length:
            raise TextTooLongError(
                f"Text contains {token_count} tokens; maximum is {self.max_length}"
            )

        return {"normalized_text": normalized, "token_count": token_count}

    @torch.inference_mode()
    def predict(self, text: str) -> dict[str, Any]:
        tokenized = self.tokenize(text)
        inputs = self.tokenizer(
            tokenized["normalized_text"],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        logits = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        result: dict[str, Any] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "token_count": tokenized["token_count"],
        }
        for target_name, target_logits in logits.items():
            labels = self.targets[target_name]
            probabilities = torch.softmax(target_logits[0].float(), dim=-1).tolist()
            best_index = int(torch.argmax(target_logits[0]).item())
            result[target_name] = {
                "label": labels[best_index],
                "probabilities": {
                    label: float(probability)
                    for label, probability in zip(labels, probabilities)
                },
            }
        return result


def default_model_dir() -> Path:
    configured = os.environ.get("MODEL_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "models" / "parsbert"

