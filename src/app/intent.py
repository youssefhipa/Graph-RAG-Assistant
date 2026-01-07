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
    "seller_count",
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
        "product_search": [
            "find",
            "show",
            "products",
            "search",
            "category",
            "electronics",
            "eletronicos",
            "perfumes",
            "perfumaria",
            "fashion",
            "beauty",
            "furniture",
            "sports",
            "toys",
            "price",
            "best",
            "top",
            "recommend",
            "rating",
        ],
        "seller_count": ["seller", "sellers", "how many sellers", "count sellers", "number of sellers"],
        "delivery_delay": ["delay", "late", "shipping", "delivery", "on time", "sla"],
        "review_sentiment": ["review", "rating", "feedback", "sentiment", "score"],
        "seller_performance": ["seller", "performance", "reliability", "fulfillment"],
        "state_trend": ["state", "city", "region", "trend", "geo", "location"],
        "category_insight": [
            "category",
            "categories",
            "category insight",
            "popular category",
            "top categories",
            "trending",
        ],
        "recommendation": ["recommend", "suggest", "best for me", "which", "choose"],
        "customer_behavior": ["repeat", "customer", "buyer", "behavior", "churn", "loyal"],
        "faq": ["what is", "how to", "when", "policy", "faq"],
    }

    def predict(self, text: str) -> IntentResult:
        lowered = text.lower()

        # Priority rules to avoid misclassifying common demo questions.
        if re.search(r"\b(how many sellers|number of sellers|count sellers)\b", lowered):
            return IntentResult(intent="seller_count", confidence=1.0, matched=["seller_count_rule"])
        if re.search(r"\b(late|delay|delivery)\b", lowered):
            return IntentResult(intent="delivery_delay", confidence=0.9, matched=["delivery_delay_rule"])
        if "seller" in lowered and re.search(r"\b(reliability|performance)\b", lowered):
            return IntentResult(intent="seller_performance", confidence=0.9, matched=["seller_performance_rule"])
        if re.search(r"\b(categories|category)\b", lowered) and re.search(
            r"\b(popular|top|trending|most)\b", lowered
        ):
            return IntentResult(intent="category_insight", confidence=0.9, matched=["category_insight_rule"])
        if re.search(r"\b(recommend|suggest)\b", lowered):
            return IntentResult(intent="recommendation", confidence=0.9, matched=["recommendation_rule"])
        if re.search(r"\b(review|reviews|sentiment|feedback)\b", lowered):
            return IntentResult(intent="review_sentiment", confidence=0.9, matched=["review_sentiment_rule"])
        if re.search(r"\bstate\b", lowered) and re.search(r"\b(most|orders|trend)\b", lowered):
            return IntentResult(intent="state_trend", confidence=0.9, matched=["state_trend_rule"])
        if re.search(r"\b(repeat|customer|buyer)\b", lowered):
            return IntentResult(intent="customer_behavior", confidence=0.8, matched=["customer_behavior_rule"])
        if re.search(r"\b(product|products|category|electronics|eletronicos|perfumes|perfumaria|top|best)\b", lowered):
            return IntentResult(intent="product_search", confidence=0.8, matched=["product_search_rule"])

        best_intent = "unknown"
        best_score = 0.0
        matched_keywords: List[str] = []

        for intent, keywords in self.KEYWORDS.items():
            hits = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered)]
            score = len(hits) / max(len(keywords), 1)
            if score > best_score:
                best_intent = intent
                best_score = score
                matched_keywords = hits

        return IntentResult(intent=best_intent, confidence=best_score, matched=matched_keywords)
