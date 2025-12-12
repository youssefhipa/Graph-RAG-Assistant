from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


INTENTS = [
    "product_search",
    "delivery_delay",
    "review_sentiment",
    "seller_performance",
    "state_trend",
    "category_insight",
    "recommendation",
    "customer_behavior",
    "faq",
    "unknown",
]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched: List[str]


class IntentClassifier:
    """
    Lightweight, rule-based intent classifier tailored to ecommerce themes.
    Replace/extend with an LLM or ML model if desired.
    """

    KEYWORDS = {
        "product_search": ["find", "show", "products", "search", "category", "price"],
        "delivery_delay": ["delay", "late", "shipping", "delivery", "on time", "sla"],
        "review_sentiment": ["review", "rating", "feedback", "sentiment", "score"],
        "seller_performance": ["seller", "performance", "reliability", "fulfillment"],
        "state_trend": ["state", "city", "region", "trend", "geo", "location"],
        "category_insight": ["category", "perfume", "electronics", "fashion", "insight"],
        "recommendation": ["recommend", "suggest", "best for me", "which", "choose"],
        "customer_behavior": ["repeat", "customer", "buyer", "behavior", "churn", "loyal"],
        "faq": ["what is", "how to", "when", "policy", "faq"],
    }

    def predict(self, text: str) -> IntentResult:
        lowered = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        matched_keywords: List[str] = []

        for intent, keywords in self.KEYWORDS.items():
            hits = [kw for kw in keywords if re.search(rf"\\b{re.escape(kw)}\\b", lowered)]
            score = len(hits) / max(len(keywords), 1)
            if score > best_score:
                best_intent = intent
                best_score = score
                matched_keywords = hits

        return IntentResult(intent=best_intent, confidence=best_score, matched=matched_keywords)
