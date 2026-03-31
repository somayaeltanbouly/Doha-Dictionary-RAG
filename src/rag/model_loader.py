"""
model_loader.py — Model loading factory for the Doha Dictionary RAG system.

``ModelLoader.load()`` returns a ready-to-use ``GenerationBackend`` instance
based on the requested model type.

Supported model types
---------------------
- ``"fanar"``  : Fanar API (OpenAI-compatible interface, model ``Fanar-C-1-8.7B``)
- ``"gemini"`` : Google Gemini via ``google-generativeai`` (model ``gemini-2.5-pro``)
- ``"hf"``     : HuggingFace ``transformers`` with fp16 or 4-bit quantization (default ALLaM)

Usage::

    from src.rag.model_loader import ModelLoader

    backend = ModelLoader.load("fanar")
    answer  = backend.generate("ما معنى كلمة الجفاء؟")

    # Override the default model
    backend = ModelLoader.load("hf", model_id="ALLaM-AI/ALLaM-7B-Instruct-preview")
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional
import torch


# ──────────────────────────────────────────────────────────────────────────── #
# Abstract base                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class GenerationBackend(ABC):
    """Interface all generation backends must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's text response for *prompt*."""


# ──────────────────────────────────────────────────────────────────────────── #
# Fanar backend                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class FanarBackend(GenerationBackend):
    """Fanar API backend using the OpenAI-compatible chat interface.

    Reads ``FANAR_API_KEY`` from the environment.

    Args:
        model: Fanar model name to call. Defaults to ``"Fanar-C-1-8.7B"``.
    """

    _BASE_URL = "https://api.fanar.qa/v1"
    _DEFAULT_MODEL = "Fanar-C-1-8.7B"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from openai import OpenAI

        api_key = os.environ.get("FANAR_API_KEY")
        if not api_key:
            raise EnvironmentError("FANAR_API_KEY environment variable is not set.")

        self._client = OpenAI(base_url=self._BASE_URL, api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        from openai import BadRequestError

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except BadRequestError as exc:
            if "content_filter" in str(exc):
                return "Skipped due to content filter."
            raise

    def __str__(self) -> str:
        return f"FanarBackend({self._model})"


# ──────────────────────────────────────────────────────────────────────────── #
# Gemini backend                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

class GeminiBackend(GenerationBackend):
    """Google Gemini backend via the ``google-generativeai`` SDK.

    Reads ``GEMINI_API_KEY`` from the environment.

    Args:
        model:        Gemini model name. Defaults to ``"gemini-2.5-pro"``.
        max_retries:  Number of retry attempts on transient failures.
        retry_delay:  Seconds to wait between retries.
    """

    _DEFAULT_MODEL = "gemini-2.5-pro"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model_name=model)
        self._model_name = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def generate(self, prompt: str) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=self._genai.types.GenerationConfig(temperature=0),
                )
                return response.text
            except Exception:
                if attempt == self._max_retries:
                    raise
                time.sleep(self._retry_delay)
        return ""  # unreachable

    def __str__(self) -> str:
        return f"GeminiBackend({self._model_name})"


# ──────────────────────────────────────────────────────────────────────────── #
# HuggingFace backend                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

class HuggingFaceBackend(GenerationBackend):
    """HuggingFace ``transformers`` backend for local inference.

    Defaults to the ALLaM instruction-tuned model loaded in fp16 mixed precision.
    Optionally enable 4-bit BitsAndBytes quantization for lower VRAM usage.

    Args:
        model_id:       HuggingFace model ID or local path.
        load_in_4bit:   Enable 4-bit BitsAndBytes quantization. Defaults to
                        ``False``; the model is loaded in fp16 by default.
        max_new_tokens: Maximum number of tokens to generate.
    """

    _DEFAULT_MODEL_ID = "ALLaM-AI/ALLaM-7B-Instruct-preview"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self._model.eval()
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def __str__(self) -> str:
        return f"HuggingFaceBackend({self._model_id})"


# ──────────────────────────────────────────────────────────────────────────── #
# Factory                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class ModelLoader:
    """Factory class that instantiates a :class:`GenerationBackend` by type name.

    Example::

        backend = ModelLoader.load("gemini")
        backend = ModelLoader.load("hf", model_id="ALLaM-AI/ALLaM-7B-Instruct-preview")
    """

    SUPPORTED = ("fanar", "gemini", "hf")

    @classmethod
    def load(
        cls,
        model_type: str,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationBackend:
        """Return a configured ``GenerationBackend`` for *model_type*.

        Args:
            model_type: ``"fanar"``, ``"gemini"``, or ``"hf"``.
            model_id:   Optional model identifier / local path. Overrides the
                        backend's default model name.
            **kwargs:   Additional arguments forwarded to the backend constructor
                        (e.g. ``max_new_tokens`` for HuggingFace).

        Returns:
            A ready-to-use :class:`GenerationBackend` instance.

        Raises:
            ValueError:       If *model_type* is not recognised.
            EnvironmentError: If the required API key is missing.
        """
        model_type = model_type.lower()

        if model_type == "fanar":
            if model_id:
                kwargs["model"] = model_id
            return FanarBackend(**kwargs)

        if model_type == "gemini":
            if model_id:
                kwargs["model"] = model_id
            return GeminiBackend(**kwargs)

        if model_type == "hf":
            if model_id:
                kwargs["model_id"] = model_id
            return HuggingFaceBackend(**kwargs)

        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Supported types: {cls.SUPPORTED}."
        )
