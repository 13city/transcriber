from __future__ import annotations


class FullStopPunctuator:
    """Punctuation restoration using the FullStop Hugging Face model.

    The wrapper package handles text chunking for the underlying token-classification
    model. We invoke it only when punctuation is sparse; Faster Whisper output often
    already contains useful punctuation that should not be needlessly rewritten.
    """

    def __init__(self, model_name: str) -> None:
        from deepmultilingualpunctuation import PunctuationModel

        self.model = PunctuationModel(model=model_name)

    def restore(self, text: str) -> str:
        return self.model.restore_punctuation(text).strip()
