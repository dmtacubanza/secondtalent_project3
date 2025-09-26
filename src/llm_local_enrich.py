# enrich_local.py (hardened)
from __future__ import annotations
from typing import Iterable, List, Dict, Any, Tuple
import re
import pandas as pd
from tqdm import tqdm
from .ml_summarizer import summarize_local
from .ml_sentiment import classify, classify_reviews
from .io_utils import conn, write_df

IN_TABLE = "silver_reviews"
OUT_TABLE = "silver_llm_outputs"                # per-review outputs (shared name; see note below)
PRODUCT_OUT_TABLE = "silver_product_sentiment"  # per-product aggregates
_ALLOWED = {"positive", "neutral", "negative"}

# ------------- helpers -------------

def _chunks(it: Iterable[Dict], size: int):
    buf = []
    for row in it:
        buf.append(row)
        if len(buf) >= size:
            yield buf; buf = []
    if buf:
        yield buf

def _label_only(sent):
    """Accept classify(...) returning either str or (label, score) and normalize to label."""
    if isinstance(sent, tuple):
        return (sent[0] or "").lower()
    return (sent or "").lower()

def _ensure_tables():
    c = conn()
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
            review_id  TEXT,
            product_id TEXT,
            summary    TEXT,
            sentiment  TEXT
        )"""
    )
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {PRODUCT_OUT_TABLE} (
            product_id      TEXT,
            sentiment       TEXT,
            confidence      DOUBLE,
            mean_pos_prob   DOUBLE,
            n_reviews       INTEGER,
            frac_pos_pred   DOUBLE,
            frac_neg_pred   DOUBLE,
            product_summary TEXT
        )"""
    )

_SENT_END_RE = re.compile(r"([.!?])\s+")
def _first_sentence(text: str) -> str:
    # quick-and-dirty first sentence extractor
    t = (text or "").strip()
    if not t:
        return ""
    m = _SENT_END_RE.split(t, maxsplit=1)
    if len(m) >= 2:
        return (m[0] + m[1]).strip()
    return t.splitlines()[0].strip()

def _fallback_summary(text: str, max_words: int = 60) -> str:
    """
    Deterministic fallback: first sentence (preferred), or first N words.
    """
    t = (text or "").strip()
    if not t:
        return ""
    first = _first_sentence(t)
    if first:
        return " ".join(first.split()[:max_words])
    return " ".join(t.split()[:max_words])

def _safe_summarize(text: str, *, max_words: int = 120) -> str:
    """
    Guard around summarize_local:
      - handles None/"" and exceptions
      - retries with smaller max_words
      - deterministic fallback
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Try the requested limit
    try:
        s = summarize_local(t, max_words=max_words) or ""
        s = s.strip()
        if s:
            return s
    except Exception:
        pass
    # Try smaller limits to avoid token/length constraints in some models
    for mw in (80, 60, 40):
        try:
            s = summarize_local(t[:2000], max_words=mw) or ""
            s = s.strip()
            if s:
                return s
        except Exception:
            continue
    # Fallback deterministic
    return _fallback_summary(t, max_words=min(max_words, 60))

# ------------- main -------------

def run(limit: int | None = None, batch_size: int = 64,
        max_reviews_per_product_for_summary: int = 30,
        truncate_chars_per_review: int = 350) -> int:
    """
    Local-only enrichment:
      - per-review summary via summarize_local (safely wrapped)
      - per-review sentiment via local ML (classify)
      - per-product sentiment via classify_reviews aggregation
      - per-product summary via summarize_local over capped concatenation
    """
    # 0) Load & validate
    q = f"SELECT review_id, product_id, review_text FROM {IN_TABLE}"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = conn().execute(q).df()
    _ensure_tables()

    if df.empty:
        # still write empty outputs to keep downstream stable
        write_df(OUT_TABLE, pd.DataFrame(columns=["review_id","product_id","summary","sentiment"]), mode="replace")
        write_df(PRODUCT_OUT_TABLE, pd.DataFrame(columns=[
            "product_id","sentiment","confidence","mean_pos_prob","n_reviews",
            "frac_pos_pred","frac_neg_pred","product_summary"
        ]), mode="replace")
        return 0

    expected = {"review_id", "product_id", "review_text"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"{IN_TABLE} is missing columns: {sorted(missing)}")

    # Normalize text and drop empty reviews early (these would yield empty summaries anyway)
    df["review_text"] = df["review_text"].astype(str).fillna("")
    before = len(df)
    df = df[df["review_text"].str.strip().ne("")]
    filtered = before - len(df)
    if filtered > 0:
        print(f"[info] filtered out {filtered} rows with empty review_text (kept {len(df)})")

    # 1) Per-review pass
    rows: List[Dict[str, Any]] = []
    for batch in _chunks(df.to_dict("records"), batch_size):
        for r in tqdm(batch, total=len(batch), leave=False):
            rid = str(r["review_id"])
            pid = str(r["product_id"])
            text = (r.get("review_text") or "").strip()

            # Safe summary
            summary = _safe_summarize(text, max_words=120)

            # Local sentiment
            try:
                lab = _label_only(classify(text))
            except Exception:
                lab = "neutral"
            if lab not in _ALLOWED:
                lab = "neutral"

            rows.append({
                "review_id": rid,
                "product_id": pid,
                "summary": summary,
                "sentiment": lab
            })

    out = pd.DataFrame(rows, columns=["review_id","product_id","summary","sentiment"])
    write_df(OUT_TABLE, out, mode="replace")

    # 2) Per-product aggregation
    # Gather per-product text lists (non-empty only)
    prod_texts = (
        df.groupby("product_id")["review_text"]
          .apply(lambda s: [t for t in s.tolist() if isinstance(t, str) and t.strip()])
          .to_dict()
    )

    product_rows: List[Dict[str, Any]] = []
    for pid, texts in tqdm(prod_texts.items(), total=len(prod_texts), leave=False):
        if not texts:
            product_rows.append({
                "product_id": str(pid),
                "sentiment": "neutral",
                "confidence": 0.0,
                "mean_pos_prob": 0.5,
                "n_reviews": 0,
                "frac_pos_pred": 0.0,
                "frac_neg_pred": 0.0,
                "product_summary": ""
            })
            continue

        # Local ML aggregation for sentiment
        try:
            label, confidence, stats = classify_reviews(
                texts,
                batch_size=32,
                pos_threshold=0.60,
                neg_threshold=0.40,
                truncate_chars=512
            )
        except Exception:
            label, confidence, stats = "neutral", 0.0, {"mean_pos_prob": 0.5, "n": len(texts), "frac_pos_pred": 0.0, "frac_neg_pred": 0.0}

        if label not in _ALLOWED:
            label = "neutral"

        # Local product paragraph (safe)
        subset = texts[:max_reviews_per_product_for_summary]
        subset = [t[:truncate_chars_per_review] for t in subset]
        reviews_block = "\n".join(subset)
        product_summary = _safe_summarize(reviews_block, max_words=120) if reviews_block.strip() else ""

        product_rows.append({
            "product_id": str(pid),
            "sentiment": label,
            "confidence": float(confidence),
            "mean_pos_prob": float(stats.get("mean_pos_prob", 0.5)),
            "n_reviews": int(stats.get("n", len(texts))),
            "frac_pos_pred": float(stats.get("frac_pos_pred", 0.0)),
            "frac_neg_pred": float(stats.get("frac_neg_pred", 0.0)),
            "product_summary": product_summary
        })

    prod_out = pd.DataFrame(product_rows, columns=[
        "product_id","sentiment","confidence","mean_pos_prob","n_reviews",
        "frac_pos_pred","frac_neg_pred","product_summary"
    ])
    write_df(PRODUCT_OUT_TABLE, prod_out, mode="replace")
    return len(out)
