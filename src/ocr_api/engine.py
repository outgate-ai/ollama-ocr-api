"""Vision-model engine wrapper: eager startup load, per-request inference."""

from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str
    prompt_tokens: int
    output_tokens: int


class Engine(Protocol):
    @property
    def state(self) -> str: ...
    @property
    def error(self) -> str | None: ...
    def load(self) -> None: ...
    def generate(
        self,
        *,
        prompt: str,
        images_b64: list[str],
        max_new_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> OcrResult: ...


class DecodedImage:
    __slots__ = ("data", "mime")

    def __init__(self, data: bytes, mime: str) -> None:
        self.data = data
        self.mime = mime


def decode_base64_image(b64: str, max_bytes: int) -> DecodedImage:
    """Decode a base64-encoded image, stripping any ``data:`` prefix."""
    payload = b64.strip()
    mime = "image/png"
    if payload.startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
        except ValueError as exc:
            raise ValueError("malformed data URL") from exc
        if ";" in header:
            mime = header[len("data:") : header.index(";")] or mime
    try:
        raw = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc
    if len(raw) > max_bytes:
        raise ValueError(
            f"image too large ({len(raw)} bytes, max {max_bytes})"
        )
    if not raw:
        raise ValueError("empty image payload")
    return DecodedImage(data=raw, mime=mime)


class VisionEngine:
    """Eagerly loads a Hugging Face image-text-to-text model at startup."""

    def __init__(
        self,
        *,
        model_id: str,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto",
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._max_image_bytes = max_image_bytes
        self._processor = None
        self._model = None
        self._state: Literal["unloaded", "loading", "ready", "error"] = "unloaded"
        self._error: str | None = None
        self._lock = Lock()
        # Serialize generate() calls — HF generate() is not thread-safe for a
        # single model instance and a single GPU can't parallelize forward passes
        # anyway without real batching.
        self._generate_lock = Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        with self._lock:
            if self._state == "ready":
                return
            self._state = "loading"
            logger.info(
                "loading vision model %s (device=%s, dtype=%s)",
                self._model_id,
                self._device,
                self._dtype,
            )
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                torch_dtype = self._resolve_torch_dtype(torch)
                device_map = "auto" if self._device == "auto" else self._device

                self._processor = AutoProcessor.from_pretrained(self._model_id)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                self._model.eval()
                self._state = "ready"
                logger.info(
                    "vision model ready (device=%s, dtype=%s)",
                    self._effective_device(),
                    torch_dtype,
                )
            except Exception as exc:
                self._state = "error"
                self._error = str(exc)
                logger.exception("failed to load vision model")
                raise

    def generate(
        self,
        *,
        prompt: str,
        images_b64: list[str],
        max_new_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> OcrResult:
        if self._state != "ready" or self._processor is None or self._model is None:
            raise RuntimeError(f"engine not ready (state={self._state})")
        if not images_b64:
            raise ValueError("at least one image is required")

        import torch
        from PIL import Image

        pil_images = []
        for b64 in images_b64:
            decoded = decode_base64_image(b64, self._max_image_bytes)
            try:
                img = Image.open(io.BytesIO(decoded.data))
                img.load()
            except Exception as exc:
                raise ValueError(f"unable to decode image: {exc}") from exc
            pil_images.append(img.convert("RGB"))

        content: list[dict] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        with self._generate_lock:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            inputs.pop("token_type_ids", None)

            gen_kwargs: dict = {"max_new_tokens": int(max_new_tokens)}
            if temperature is not None and temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)
            else:
                gen_kwargs["do_sample"] = False
            if top_p is not None and 0 < top_p <= 1:
                gen_kwargs["top_p"] = float(top_p)
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = int(top_k)

            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **gen_kwargs)

            output_ids = generated_ids[0][prompt_len:]
            text = self._processor.decode(
                output_ids, skip_special_tokens=True
            )
            output_tokens = int(output_ids.shape[0])

        return OcrResult(text=text, prompt_tokens=prompt_len, output_tokens=output_tokens)

    def _effective_device(self) -> str:
        if self._model is None:
            return self._device
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return self._device

    def _resolve_torch_dtype(self, torch_mod):
        if self._dtype == "auto":
            return "auto"
        mapping = {
            "bfloat16": torch_mod.bfloat16,
            "float16": torch_mod.float16,
            "float32": torch_mod.float32,
        }
        return mapping[self._dtype]
