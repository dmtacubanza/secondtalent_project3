from pydantic import BaseModel
from typing import Optional

class Review(BaseModel):
    review_id: str
    product_id: str
    user_id: Optional[str] = None
    profile_name: Optional[str] = None
    review_text: str  # from Text
    rating: float     # from Score (1–5)
    review_time: Optional[str] = None  # epoch seconds -> ISO timestamp
    summary_title: Optional[str] = None  # from Summary
    helpful_numer: Optional[int] = None
    helpful_denom: Optional[int] = None
    helpful_ratio: Optional[float] = None

class LLMReviewResult(BaseModel):
    product_id: str
    review_id: str
    summary: str
    sentiment: str  # positive|neutral|negative

class ProductAggregate(BaseModel):
    product_id: str
    product_name: Optional[str]
    avg_rating: float
    sentiment: str
    narrative_summary: str

from pydantic import BaseModel
from typing import Optional

class Review(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    review_text: str
    rating: float
    review_date: Optional[str] = None

class LLMReviewResult(BaseModel):
    product_id: str
    review_id: str
    summary: str
    sentiment: str  # positive|neutral|negative

class ProductAggregate(BaseModel):
    product_id: str
    product_name: Optional[str]
    avg_rating: float
    sentiment: str
    narrative_summary: str