from functools import lru_cache
from typing import Iterable, List, Tuple, Dict, Sequence, Optional
from transformers import pipeline

@lru_cache(maxsize=1)
def _clf():
    # DistilBERT SST-2 (binary pos/neg)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def _as_pos_prob(label: str, score: float) -> float:
    """Convert SST-2 output (predicted label + its score) to P(positive)."""
    # Hugging Face returns score for the predicted class only.
    # If prediction is POSITIVE: P(pos)=score, else P(pos)=1-score.
    return score if label.upper() == "POSITIVE" else (1.0 - score)

def classify(text: str) -> Tuple[str, float]:
    """
    Backwards-compatible single-text classifier.
    Returns (label, confidence). Label in {'positive','neutral','negative'}.
    """
    if not text or not text.strip():
        return "neutral", 0.0
    res = _clf()(text[:512])[0]
    label = res["label"].lower()  # 'positive' or 'negative'
    score = float(res["score"])
    
    # Create a neutral band for uncertain predictions
    if score < 0.6:
        return "neutral", score
    return label, score

def classify_reviews(
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    pos_threshold: float = 0.60,
    neg_threshold: float = 0.40,
    truncate_chars: int = 512,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Aggregate sentiment across multiple reviews.
    Returns (label, confidence, stats) where:
      - label ∈ {'positive','neutral','negative'}
      - confidence ∈ [0,1] reflects distance from neutrality
      - stats contains helpful aggregates (mean_pos_prob, n, frac_pos_pred, frac_neg_pred)

    Heuristic:
      - Compute per-review P(positive)
      - mean_pos_prob >= pos_threshold → 'positive'
      - mean_pos_prob <= neg_threshold → 'negative'
      - otherwise → 'neutral'
    """
    # Clean inputs
    texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    if not texts:
        return "neutral", 0.0, {"mean_pos_prob": 0.5, "n": 0, "frac_pos_pred": 0.0, "frac_neg_pred": 0.0}

    clf = _clf()
    # Batch inference
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:truncate_chars] for t in texts[i:i+batch_size]]
        results.extend(clf(batch, truncation=True))

    # Convert to positive probabilities
    pos_probs: List[float] = []
    pos_preds = 0
    neg_preds = 0
    for r in results:
        label = str(r["label"])
        score = float(r["score"])
        pos_probs.append(_as_pos_prob(label, score))
        if label.upper() == "POSITIVE":
            pos_preds += 1
        else:
            neg_preds += 1

    mean_pos = sum(pos_probs) / len(pos_probs)
    # Decide label based on mean probability with neutral band
    if mean_pos >= pos_threshold:
        label = "positive"
    elif mean_pos <= neg_threshold:
        label = "negative"
    else:
        label = "neutral"

    # Confidence: distance from 0.5 scaled to [0,1]
    confidence = min(1.0, max(0.0, abs(mean_pos - 0.5) * 2.0))

    stats = {
        "mean_pos_prob": mean_pos,
        "n": len(texts),
        "frac_pos_pred": pos_preds / len(texts),
        "frac_neg_pred": neg_preds / len(texts),
    }
    return label, confidence, stats
