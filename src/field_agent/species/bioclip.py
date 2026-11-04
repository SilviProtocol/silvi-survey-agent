"""One small local photo scorer: BioCLIP 2 against a plain name list.

The score for one photo is a softmax over the model's image-text
similarities for the names in `fixtures/site-species-yard.json`.
Weights come from a local path the user gives (`BIOCLIP2_WEIGHTS`, a
directory holding `open_clip_model.safetensors` and its config, or a
single checkpoint file). The stack is built once per process and can
be warmed at startup; when it cannot run, the reason is logged once
and every photo gets an honest empty result.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

_log = logging.getLogger("field_agent.species")

_STACK: dict = {"built": False, "scorer": None, "reason": None}


def candidates_path(start: Path | None = None) -> Path:
    """The plain name list, searched upward from this file."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "fixtures" / "site-species-yard.json"
        if candidate.is_file():
            return candidate
    return Path("fixtures") / "site-species-yard.json"


def load_names(path: Path | None = None) -> list[str]:
    data = json.loads((path or candidates_path()).read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else data
    return [str(n) for n in names if str(n).strip()]


class _Scorer:
    """open_clip loaded once; texts encoded once."""

    def __init__(self, weights: str, names: list[str], device: str = "cpu"):
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                f"missing {exc.name}; install the species extra (uv sync --extra species)"
            ) from exc
        if not Path(weights).exists():
            raise RuntimeError(f"no BioCLIP weights at {weights} (set BIOCLIP2_WEIGHTS)")
        self._torch = torch
        self._image = Image
        self.names = names
        model_ref = f"local-dir:{weights}" if Path(weights).is_dir() else weights
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_ref)
            tokenizer = open_clip.get_tokenizer(model_ref)
        except Exception as exc:
            raise RuntimeError(f"could not load BioCLIP weights at {weights}: {exc}") from exc
        self._model = self._model.to(device).eval()
        self._device = device
        with torch.no_grad():
            self._logit_scale = float(self._model.logit_scale.exp().item())
            texts = tokenizer([f"a photo of {name}." for name in names]).to(device)
            feats = self._model.encode_text(texts)
            self._text_feats = feats / feats.norm(dim=-1, keepdim=True)

    def score(self, photo: bytes) -> list[dict]:
        torch = self._torch
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(photo)
            path = Path(handle.name)
        try:
            image = self._preprocess(self._image.open(path).convert("RGB")).unsqueeze(0)
        finally:
            path.unlink(missing_ok=True)
        with torch.no_grad():
            feats = self._model.encode_image(image.to(self._device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (self._logit_scale * feats @ self._text_feats.T)[0].tolist()
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        total = sum(exps) or 1.0
        ranked = sorted(
            ({"name": name, "rank": "species", "p": e / total, "common_name": None}
             for name, e in zip(self.names, exps, strict=True)),
            key=lambda c: -c["p"],
        )
        return ranked[:5]


def _build_scorer() -> _Scorer:
    names = load_names()
    if not names:
        raise RuntimeError("the species name list is empty")
    weights = os.environ.get("BIOCLIP2_WEIGHTS", "")
    if not weights:
        raise RuntimeError("BIOCLIP2_WEIGHTS is not set")
    return _Scorer(weights, names)


def get_scorer():
    """The cached scorer, or None; the reason is logged once."""
    if not _STACK["built"]:
        _STACK["built"] = True
        try:
            _STACK["scorer"] = _build_scorer()
        except Exception as exc:  # noqa: BLE001 - reported once, then honest empties
            _STACK["reason"] = str(exc)
            _log.warning("species: off, %s", exc)
    return _STACK["scorer"]


def bioclip_status() -> tuple[bool, str]:
    ready = get_scorer() is not None
    return ready, "" if ready else str(_STACK.get("reason") or "unavailable")


def score_photo(photo: bytes) -> list[dict]:
    scorer = get_scorer()
    if scorer is None:
        return []
    try:
        return scorer.score(photo)
    except Exception:  # noqa: BLE001 - one bad photo is an empty answer
        return []


def _reset_for_tests() -> None:
    _STACK.update(built=False, scorer=None, reason=None)
