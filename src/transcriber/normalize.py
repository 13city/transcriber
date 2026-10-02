from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .text import split_words_for_llm

SYSTEM_PROMPT = """You are a conservative transcript editor for academic lectures.
Your task is normalization, not summarization or rewriting.

Rules:
1. Preserve every substantive claim, qualification, example, number, unit, and sequence of ideas.
2. Do not summarize, shorten, expand, teach, fact-check, or add outside knowledge.
3. Correct capitalization, punctuation, spacing, and obvious grammatical artifacts from speech recognition.
4. Join sentence fragments when they clearly belong to the same sentence.
5. Split run-on text into complete sentences and coherent paragraphs.
6. Use paragraph breaks when the lecturer changes topic or begins a distinct explanation. Aim for readable paragraphs, usually 2-6 sentences.
7. Remove only obvious accidental ASR duplication, especially repeated adjacent phrases created at segment boundaries.
8. Correct an ASR word only when the intended wording is strongly supported by the immediate context. If uncertain, preserve the wording rather than guessing.
9. Preserve technical notation and terminology such as VO2 max, VT1, VT2, RER, watts, RPM, mmol/L, names, and numeric values.
10. Preserve the lecturer's order and meaning. Do not add headings unless a heading is explicitly spoken in the source.
11. Return only the normalized transcript text. Do not explain your edits.
"""


@dataclass
class NormalizationResult:
    text: str
    input_words: int
    output_words: int
    length_ratio: float
    warnings: list[str]


class QwenNormalizer:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        chunk_words: int = 650,
    ) -> None:
        self.model_name = model_name
        self.chunk_words = chunk_words
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
        )

    def _normalize_chunk(self, chunk: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ]
        templated = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        max_new_tokens = min(max(512, int(input_tokens * 1.4)), 2400)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.03,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][input_tokens:]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
        text = re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", text).strip()
        return text

    def normalize(self, text: str) -> NormalizationResult:
        chunks = split_words_for_llm(text, self.chunk_words)
        normalized = [self._normalize_chunk(chunk) for chunk in chunks]
        output = "\n\n".join(part for part in normalized if part).strip()

        input_words = len(text.split())
        output_words = len(output.split())
        ratio = output_words / max(1, input_words)
        warnings: list[str] = []
        if ratio < 0.78:
            warnings.append(
                "Normalized transcript is substantially shorter than the source; "
                "review for omitted content."
            )
        elif ratio > 1.22:
            warnings.append(
                "Normalized transcript is substantially longer than the source; "
                "review for added content."
            )
        return NormalizationResult(
            output,
            input_words,
            output_words,
            ratio,
            warnings,
        )
