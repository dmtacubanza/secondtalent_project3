import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional
from .config import CFG

_CACHE = Path(CFG.cache_dir)
_CACHE.mkdir(exist_ok=True)

def _word_trim(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()

def _local_sum_cache_key(text: str, max_words: int) -> Path:
    import hashlib
    h = hashlib.sha256(f"{max_words}|{text}".encode()).hexdigest()
    return _CACHE / f"local_summary_{h}.json"

def _local_sum_get(text: str, max_words: int) -> Optional[str]:
    p = _local_sum_cache_key(text, max_words)
    if p.exists():
        try:
            return json.loads(p.read_text())["summary"]
        except Exception:
            return None
    return None

def _local_sum_put(text: str, max_words: int, summary: str):
    _local_sum_cache_key(text, max_words).write_text(json.dumps({"summary": summary}))

@lru_cache(maxsize=1)
def _hf_summarizer():
    """
    Lazily create a transformers summarization pipeline.
    Override the model with CFG.local_summary_model if desired.
    """
    try:
        from transformers import pipeline
        model_name = getattr(CFG, "local_summary_model", "sshleifer/distilbart-cnn-12-6")
        return pipeline("summarization", model=model_name)
    except Exception:
        return None

def _summarize_with_sumy(text: str, max_words: int) -> Optional[str]:
    """Extractive fallback via sumy TextRank if installed."""
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        if not text or not text.strip():
            return ""
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sentences = max(2, min(7, max_words // 20))
        sents = summarizer(parser.document, sentences)
        return _word_trim(" ".join(str(s) for s in sents), max_words)
    except Exception:
        return None

def summarize_local(text: str, max_words: int = 120) -> str:
    """
    Tiered local summary:
      1) transformers abstractive
      2) sumy text-rank (if installed)
      3) heuristic (first ~2 sentences), trimmed to max_words
    """
    if not text or not text.strip():
        return ""

    cached = _local_sum_get(text, max_words)
    if cached:
        return cached

    # 1) transformers
    try:
        hf = _hf_summarizer()
        if hf is not None:
            max_tokens = int(max(56, min(400, max_words * 1.6)))
            min_tokens = max(24, int(max_tokens * 0.35))
            out = hf(text, truncation=True, max_length=max_tokens, min_length=min_tokens, do_sample=False)
            if out:
                s = _word_trim(out[0]["summary_text"], max_words)
                _local_sum_put(text, max_words, s)
                return s
    except Exception:
        pass

    # 2) sumy extractive
    s2 = _summarize_with_sumy(text, max_words)
    if s2:
        _local_sum_put(text, max_words, s2)
        return s2

    # 3) heuristic
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    s3 = _word_trim(" ".join(sents[:2]) if len(sents) >= 2 else text, max_words)
    _local_sum_put(text, max_words, s3)
    return s3
