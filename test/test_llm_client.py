from ..src.llm_client import LLMClient
from ..src.prompts import REVIEW_PROMPT

def test_client_prompt():
    c = LLMClient()
    # minimal sanity check
    print("Provider:", c.provider)

    # very short review to test
    review = "Great mouse. Tracks well on any surface but the scroll wheel is noisy."
    prompt = REVIEW_PROMPT.format(review=review)

    # if your LLMClient has complete_json
    resp = c.complete_json(prompt)
    print("Raw response:", resp)

    # optional: basic asserts
    assert isinstance(resp, dict)
    assert "summary" in resp
    assert resp.get("sentiment") in {"positive", "neutral", "negative"}
    print("Summary:", resp["summary"])
    print("Sentiment:", resp["sentiment"])

if __name__ == "__main__":
    test_client_prompt()
