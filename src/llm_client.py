# llm/remote_client.py
import json
import time
from pathlib import Path
from typing import Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import CFG

_CACHE = Path(CFG.cache_dir)
_CACHE.mkdir(exist_ok=True)

class LLMClient:
    """
    Handles remote providers (OpenAI, Gemini, HuggingFace) with:
      - complete_json(prompt) -> Dict[str, Any]
      - complete_text(prompt) -> str
    Includes simple file caching keyed by prompt to reduce calls.
    """

    def __init__(self):
        self.provider = CFG.llm_provider
        self.session = httpx.Client(timeout=60)

    # ---------------- cache helpers ----------------
    def _cache_key(self, prompt: str) -> Path:
        import hashlib
        return _CACHE / (hashlib.sha256(prompt.encode()).hexdigest() + ".json")

    def _cached(self, prompt: str):
        p = self._cache_key(prompt)
        return p.read_text() if p.exists() else None

    def _save_cache(self, prompt: str, content: str):
        self._cache_key(prompt).write_text(content)

    # ---------------- remote completions ----------------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def complete_json(self, prompt: str) -> Dict[str, Any]:
        cached = self._cached(prompt)
        if cached:
            # cached content for JSON is stored as raw dictionary
            return json.loads(cached)

        # --- provider-specific call returning a TEXT response as JSON ---
        if self.provider == "openai":
            payload = {
                "model": CFG.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": CFG.temperature,
                "max_tokens": CFG.max_tokens,
            }
            headers = {"Authorization": f"Bearer {CFG.openai_api_key}"}
            r = self.session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{CFG.gemini_model}:generateContent?key={CFG.gemini_api_key}"
            r = self.session.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
            r.raise_for_status()
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"]

        else:  # Hugging Face
            headers = {"Authorization": f"Bearer {CFG.hf_api_key}"}
            r = self.session.post(
                "https://api-inference.huggingface.co/models/" + CFG.hf_model,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": CFG.max_tokens, "temperature": CFG.temperature}},
            )
            r.raise_for_status()
            content = r.json()[0]["generated_text"]

        # --- parse JSON (with fallback extraction) ---
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])

        self._save_cache(prompt, json.dumps(data))
        time.sleep(1.0 / max(CFG.qps, 0.1))
        return data

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def complete_text(self, prompt: str) -> str:
        cached = self._cached(prompt)
        if cached:
            # text completions are cached as {"text": "..."}
            try:
                return json.loads(cached)["text"]
            except Exception:
                # if cached was JSON for the same prompt, just regenerate
                pass

        if self.provider == "openai":
            payload = {
                "model": CFG.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": CFG.temperature,
                "max_tokens": CFG.max_tokens,
            }
            headers = {"Authorization": f"Bearer {CFG.openai_api_key}"}
            r = self.session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{CFG.gemini_model}:generateContent?key={CFG.gemini_api_key}"
            r = self.session.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
            r.raise_for_status()
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"]

        else:
            headers = {"Authorization": f"Bearer {CFG.hf_api_key}"}
            r = self.session.post(
                "https://api-inference.huggingface.co/models/" + CFG.hf_model,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": CFG.max_tokens, "temperature": CFG.temperature}},
            )
            r.raise_for_status()
            content = r.json()[0]["generated_text"]

        self._save_cache(prompt, json.dumps({"text": content}))
        return content
