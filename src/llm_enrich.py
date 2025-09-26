# enrich_remote.py (hardened)
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Any
import json
import math
import re
import time
import pandas as pd
from tqdm import tqdm
from .prompts import REVIEW_PROMPT, PRODUCT_PROMPT
from .llm_client import LLMClient
from .io_utils import conn, write_df

IN_TABLE = "silver_reviews"
OUT_TABLE = "silver_llm_outputs"                # per-review outputs
PRODUCT_OUT_TABLE = "silver_product_sentiment"  # per-product aggregates
_ALLOWED = {"positive", "neutral", "negative"}

_POS_WEIGHT = 1.0
_NEU_WEIGHT = 0.5
_NEG_WEIGHT = 0.0

# --------- helpers ---------

def _chunks(it: Iterable[Dict], size: int):
    buf = []
    for row in it:
        buf.append(row)
        if len(buf) >= size:
            yield buf; buf = []
    if buf:
        yield buf

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

def _sent_to_posprob(sent: str) -> float:
    s = (sent or "").lower()
    if s == "positive": return _POS_WEIGHT
    if s == "negative": return _NEG_WEIGHT
    return _NEU_WEIGHT

_JSON_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", flags=re.DOTALL | re.IGNORECASE)

def _extract_json(text: str) -> str:
    """
    Try hard to extract a JSON object from messy model output.
    """
    if not text:
        raise ValueError("empty model output")
    # 1) try direct parse
    t = text.strip()
    try:
        json.loads(t)
        return t
    except Exception:
        pass
    # 2) try code-fence block
    m = _JSON_FENCE_RE.search(t)
    if m:
        inner = m.group(1).strip()
        try:
            json.loads(inner)
            return inner
        except Exception:
            pass
    # 3) find first {...} span
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = t[first:last+1]
        # optional: strip trailing junk like trailing commas
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    # error
    raise ValueError("could not extract JSON from model output")

def _retry(fn, *, tries=3, base_delay=0.8, factor=1.8, on_error=None):
    """
    Simple exponential backoff retry.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if on_error:
                on_error(e, attempt)
            if attempt >= tries:
                raise
            time.sleep(base_delay * (factor ** (attempt - 1)))

def _call_review_json(client: LLMClient, review_text: str) -> Tuple[str, str]:
    """
    Returns (summary, sentiment). Sentiment defaults to 'neutral' if not valid.
    """
    # You already truncate in run() for the product summary; for per-review keep it tight too:
    prompt = REVIEW_PROMPT.format(review=review_text[:2000])

    def _do():
        # Prefer JSON endpoint; if it fails, fall back to text + parse.
        try:
            resp = client.complete_json(prompt)
            if isinstance(resp, dict):
                return resp
            # Some clients return str; parse if so
            return json.loads(_extract_json(str(resp)))
        except Exception:
            # Fallback: text completion then parse
            text = client.complete_text(prompt)
            return json.loads(_extract_json(text))

    raw = _retry(
        _do,
        tries=3,
        on_error=lambda e, a: print(f"[warn] review JSON attempt {a} failed: {e}")
    )

    summary = (raw.get("summary") or "").strip()
    sent = (raw.get("sentiment") or "").strip().lower()
    if sent not in _ALLOWED:
        sent = "neutral"
    return summary, sent

def _call_product_summary(client: LLMClient, reviews_block: str) -> str:
    prompt = PRODUCT_PROMPT.format(reviews=reviews_block)
    def _do():
        return client.complete_text(prompt)
    txt = _retry(
        _do,
        tries=3,
        on_error=lambda e, a: print(f"[warn] product summary attempt {a} failed: {e}")
    )
    return (txt or "").strip()

# --------- main ---------

def run(limit: int | None = None, batch_size: int = 64,
        max_reviews_per_product_for_summary: int = 30,
        truncate_chars_per_review: int = 350,
        pos_threshold: float = 0.60,
        neg_threshold: float = 0.40) -> int:
    """
    Remote-only enrichment:
      - per-review summary & sentiment from REVIEW_PROMPT (JSON)
      - per-product sentiment aggregated from per-review LLM sentiments
      - per-product summary from PRODUCT_PROMPT (plain text)
    """
    # 0) Pull input and check schema/rows
    q = f"SELECT review_id, product_id, review_text FROM {IN_TABLE}"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = conn().execute(q).df()

    _ensure_tables()
    if df.empty:
        print("[info] no input rows in silver_reviews")
        # check empty outputs exist
        write_df(OUT_TABLE, pd.DataFrame(columns=["review_id","product_id","summary","sentiment"]), mode="replace")
        write_df(PRODUCT_OUT_TABLE, pd.DataFrame(columns=[
            "product_id","sentiment","confidence","mean_pos_prob","n_reviews",
            "frac_pos_pred","frac_neg_pred","product_summary"
        ]), mode="replace")
        return 0

    expected_cols = {"review_id", "product_id", "review_text"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Input table {IN_TABLE} missing columns: {sorted(missing)}")

    # Drop completely empty texts
    before = len(df)
    df["review_text"] = df["review_text"].astype(str).fillna("")
    df = df[df["review_text"].str.strip().ne("")]
    after = len(df)
    if after < before:
        print(f"[info] filtered out {before - after} rows with empty review_text")

    client = LLMClient()

    # 1) Per-review pass
    per_review_rows: List[Dict[str, Any]] = []
    if df.empty:
        print("[warn] all reviews were empty; writing empty outputs")
    else:
        for batch in _chunks(df.to_dict("records"), batch_size):
            for r in tqdm(batch, total=len(batch), leave=False):
                rid = str(r["review_id"])
                pid = str(r["product_id"])
                text = (r.get("review_text") or "").strip()

                summary, sentiment = "", "neutral"
                try:
                    summary, sentiment = _call_review_json(client, text)
                except Exception as e:
                    # Keep going; log for diagnosis
                    print(f"[warn] failed to enrich review_id={rid}: {e}")

                # Optional: capture the raw payload for debugging; add a column if needed
                per_review_rows.append({
                    "review_id": rid,
                    "product_id": pid,
                    "summary": summary,
                    "sentiment": sentiment
                })

    out = pd.DataFrame(per_review_rows, columns=["review_id","product_id","summary","sentiment"])
    write_df(OUT_TABLE, out, mode="replace")

    # 2) Per-product aggregation
    if out.empty:
        write_df(PRODUCT_OUT_TABLE, pd.DataFrame([], columns=[
            "product_id","sentiment","confidence","mean_pos_prob","n_reviews",
            "frac_pos_pred","frac_neg_pred","product_summary"
        ]), mode="replace")
        return 0

    product_rows: List[Dict[str, Any]] = []
    grouped = out.groupby("product_id", dropna=False)

    for pid, grp in tqdm(grouped, total=len(grouped), leave=False):
        sents = grp["sentiment"].fillna("neutral").astype(str).str.lower().tolist()
        n = len(sents)
        pos = sum(1 for x in sents if x == "positive")
        neg = sum(1 for x in sents if x == "negative")
        neu = n - pos - neg

        mean_pos_prob = (
            pos * _POS_WEIGHT + neu * _NEU_WEIGHT + neg * _NEG_WEIGHT
        ) / max(n, 1)

        if mean_pos_prob >= pos_threshold:
            prod_label = "positive"
        elif mean_pos_prob <= neg_threshold:
            prod_label = "negative"
        else:
            prod_label = "neutral"

        confidence = min(1.0, max(0.0, abs(mean_pos_prob - 0.5) * 2.0))

        # Build product summary input
        texts = (
            df.loc[df["product_id"] == pid, "review_text"]
              .dropna()
              .astype(str)
              .tolist()
        )
        subset = texts[:max_reviews_per_product_for_summary]
        subset = [t[:truncate_chars_per_review] for t in subset]
        reviews_block = "\n".join(subset)

        product_summary = ""
        if reviews_block.strip():
            try:
                product_summary = _call_product_summary(client, reviews_block)
            except Exception as e:
                print(f"[warn] product summary failed for product_id={pid}: {e}")

        product_rows.append({
            "product_id": str(pid),
            "sentiment": prod_label,
            "confidence": float(confidence),
            "mean_pos_prob": float(mean_pos_prob),
            "n_reviews": int(n),
            "frac_pos_pred": float(pos / n if n else 0.0),
            "frac_neg_pred": float(neg / n if n else 0.0),
            "product_summary": product_summary
        })

    prod_out = pd.DataFrame(product_rows, columns=[
        "product_id","sentiment","confidence","mean_pos_prob","n_reviews",
        "frac_pos_pred","frac_neg_pred","product_summary"
    ])
    write_df(PRODUCT_OUT_TABLE, prod_out, mode="replace")
    return len(out)
