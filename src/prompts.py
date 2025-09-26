REVIEW_PROMPT = (
    "You are an expert product review analyst. Read the review below and return ONLY JSON with two keys: "
    "'summary' and 'sentiment'. The summary should briefly describe the main points. "
    "The sentiment must be exactly one of: positive, neutral, negative. "
    "Use 'positive' if the overall tone is clearly favorable, 'negative' if mostly complaints, "
    "and 'neutral' if mixed/unclear. REVIEW: {review}"
)

PRODUCT_PROMPT = (
    "You are a helpful analyst. Summarize common themes, pros, and cons across these reviews "
    "for a single product. Provide ONE paragraph (<= 120 words). Reply ONLY with the paragraph."
    "REVIEWS (one per line): {reviews}"
)