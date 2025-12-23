from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


STATES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]

COMMON_CATEGORIES = [
    "electronics",
    "perfumes",
    "perfumaria",
    "fashion",
    "beauty",
    "furniture",
    "sports",
    "toys",
]


@dataclass
class EntityResult:
    category: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    min_rating: Optional[float] = None
    min_reliability: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    product: Optional[str] = None
    seller: Optional[str] = None

    def to_params(self) -> Dict[str, object]:
        # Return all parameters (including None) so Cypher queries with optional params
        # always receive bound variables and don't error with ParameterMissing.
        return dict(self.__dict__)


class EntityExtractor:
    """
    Simple regex/string matcher. Replace with spaCy/LLM NER for higher recall.
    """

    def __init__(self, known_cities: Optional[List[str]] = None):
        self.known_cities = [c.lower() for c in known_cities] if known_cities else []

    def parse(self, text: str) -> EntityResult:
        lowered = text.lower()
        category = self._extract_category(lowered)
        state = self._extract_state(lowered)
        city = self._extract_city(lowered)
        min_rating = self._extract_rating(lowered)
        min_reliability = self._extract_reliability(lowered)
        start_date, end_date = self._extract_dates(lowered)
        product = self._extract_name(lowered, label="product")
        seller = self._extract_name(lowered, label="seller")

        return EntityResult(
            category=category,
            state=state,
            city=city,
            min_rating=min_rating,
            min_reliability=min_reliability,
            start_date=start_date,
            end_date=end_date,
            product=product,
            seller=seller,
        )

    def _extract_category(self, text: str) -> Optional[str]:
        for cat in COMMON_CATEGORIES:
            if cat in text:
                return cat
        match = re.search(r"category\s+(\w+)", text)
        if match:
            return match.group(1)
        return None

    def _extract_state(self, text: str) -> Optional[str]:
        for st in STATES:
            if re.search(rf"\b{st.lower()}\b", text):
                return st
        return None

    def _extract_city(self, text: str) -> Optional[str]:
        for city in self.known_cities:
            if city in text:
                return city.title()
        match = re.search(r"in\s+([a-zA-Z\s]+?)(?:\s+(?:with|rating|for|on|by)\b|$)", text)
        if match:
            return match.group(1).strip().title()
        return None

    def _extract_rating(self, text: str) -> Optional[float]:
        match = re.search(r"rating[s]?\s*(?:above|>|>=)?\s*(\d(?:\.\d)?)", text)
        if match:
            return float(match.group(1))
        return None

    def _extract_reliability(self, text: str) -> Optional[float]:
        match = re.search(r"(reliability|on[-\s]?time)\s*(?:above|>|>=)?\s*(0\.\d+|1\.0)", text)
        if match:
            return float(match.group(2))
        return None

    def _extract_dates(self, text: str) -> tuple[Optional[str], Optional[str]]:
        date_range = re.search(r"(20\d{2}-\d{2}-\d{2}).*?(20\d{2}-\d{2}-\d{2})", text)
        if date_range:
            return date_range.group(1), date_range.group(2)
        return None, None

    def _extract_name(self, text: str, label: str) -> Optional[str]:
        match = re.search(rf"{label}\s+([\w\s]+)", text)
        if match:
            return match.group(1).strip()
        return None
