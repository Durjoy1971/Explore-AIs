"""Backbone-specific preparation of text views before tokenization."""

BANGLABERT_MODEL = "csebuetnlp/banglabert"
BANGLABERT_NORMALIZER_COMMIT = "d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9"
BANGLABERT_NORMALIZER_REQUIREMENT = (
    "normalizer @ git+https://github.com/csebuetnlp/normalizer@" + BANGLABERT_NORMALIZER_COMMIT
)


def prepare_tokenizer_texts(texts: list[str], model_name: str) -> list[str]:
    """Apply BanglaBERT's published normalization to the already selected view."""
    if model_name != BANGLABERT_MODEL:
        return texts
    try:
        from normalizer import normalize
    except ImportError as exc:
        raise RuntimeError(
            f"BanglaBERT requires the authors' text normalizer. Install {BANGLABERT_NORMALIZER_REQUIREMENT!r}."
        ) from exc
    return [normalize(text) for text in texts]
