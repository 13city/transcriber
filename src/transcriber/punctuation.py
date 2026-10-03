from __future__ import annotations

import re
from collections.abc import Iterator

from transformers import pipeline


class FullStopPunctuator:
    """Restore punctuation with the FullStop token-classification model.

    This implementation intentionally talks to Transformers directly instead of
    depending on the older deepmultilingualpunctuation PyPI wrapper. Newer
    Transformers releases removed the deprecated grouped_entities argument used
    by older wrapper releases.
    """

    def __init__(self, model_name: str) -> None:
        # Run punctuation restoration on CPU. It is a relatively small model and
        # this avoids unnecessary CUDA probing/driver coupling on mixed systems.
        self.pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="none",
            device=-1,
        )

    @staticmethod
    def preprocess(text: str) -> list[str]:
        # Remove punctuation markers except periods/commas embedded in numbers.
        cleaned = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "", text)
        return cleaned.split()

    @staticmethod
    def _overlap_chunks(
        words: list[str],
        size: int,
        overlap: int = 0,
    ) -> Iterator[list[str]]:
        step = size - overlap
        for index in range(0, len(words), step):
            yield words[index : index + size]

    def predict(
        self,
        words: list[str],
        chunk_size: int = 230,
    ) -> list[tuple[str, str, float]]:
        if not words:
            return []

        overlap = 0 if len(words) <= chunk_size else 5
        batches = list(self._overlap_chunks(words, chunk_size, overlap))
        if batches and len(batches[-1]) <= overlap:
            batches.pop()

        tagged: list[tuple[str, str, float]] = []
        for batch_index, batch in enumerate(batches):
            active_overlap = 0 if batch_index == len(batches) - 1 else overlap
            text = " ".join(batch)
            result = self.pipe(text)
            if not result:
                tagged.extend((word, "0", 0.0) for word in batch[: len(batch) - active_overlap])
                continue

            if int(result[-1]["end"]) != len(text):
                raise RuntimeError(
                    "FullStop input chunk was clipped by the tokenizer; reduce chunk size"
                )

            char_index = 0
            result_index = 0
            limit = len(batch) - active_overlap
            for word in batch[:limit]:
                char_index += len(word) + 1
                label = "0"
                score = 0.0
                while (
                    result_index < len(result)
                    and char_index > int(result[result_index]["end"])
                ):
                    label = str(result[result_index]["entity"])
                    score = float(result[result_index]["score"])
                    result_index += 1
                tagged.append((word, label, score))

        if len(tagged) != len(words):
            raise RuntimeError(
                f"FullStop token alignment failed: expected {len(words)} words, "
                f"received {len(tagged)}"
            )
        return tagged

    @staticmethod
    def prediction_to_text(
        prediction: list[tuple[str, str, float]],
    ) -> str:
        parts: list[str] = []
        for word, label, _score in prediction:
            if label in ".,?-:":
                parts.append(word + label)
            else:
                parts.append(word)
        return " ".join(parts).strip()

    def restore(self, text: str, chunk_size: int = 230) -> str:
        words = self.preprocess(text)
        prediction = self.predict(words, chunk_size=chunk_size)
        return self.prediction_to_text(prediction)
