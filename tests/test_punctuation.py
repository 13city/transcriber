from transcriber.punctuation import FullStopPunctuator


def test_preprocess_preserves_decimal_punctuation() -> None:
    text = "RER was 0.95, then 1.0. Is that correct?"
    words = FullStopPunctuator.preprocess(text)
    assert words == ["RER", "was", "0.95", "then", "1.0", "Is", "that", "correct"]


def test_prediction_to_text() -> None:
    prediction = [
        ("Hello", ",", 0.9),
        ("world", ".", 0.9),
        ("How", "0", 0.9),
        ("are", "0", 0.9),
        ("you", "?", 0.9),
    ]
    assert (
        FullStopPunctuator.prediction_to_text(prediction)
        == "Hello, world. How are you?"
    )
