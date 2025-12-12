from __future__ import annotations

from typing import Any, Dict, Optional

from .entities import EntityResult


Query = Dict[str, Any]


QUERY_LIBRARY: Dict[str, str] = {
    "seller_count": """
    MATCH (s:Seller)
    RETURN count(s) AS seller_count
    """,
    "product_search": """
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (p)<-[:SELLS]-(s:Seller)
    WHERE ($category IS NULL OR c.name CONTAINS $category)
      AND ($state IS NULL OR s.state = $state OR p.state = $state)
      AND ($city IS NULL OR toLower(s.city) CONTAINS toLower($city))
      AND ($min_rating IS NULL OR p.avg_rating >= $min_rating)
    RETURN p.id AS id, p.name AS name, p.category AS category, p.price AS price,
           p.avg_rating AS rating, s.name AS seller, s.state AS seller_state
    ORDER BY rating DESC
    LIMIT 15
    """,
    "delivery_delay": """
    MATCH (o:Order)-[:SOLD_BY]->(s:Seller)
    WHERE ($state IS NULL OR s.state = $state)
      AND ($start_date IS NULL OR o.purchase_date >= date($start_date))
      AND ($end_date IS NULL OR o.purchase_date <= date($end_date))
    WITH o, s,
         duration.inDays(o.estimated_delivery_date, o.delivery_date).days AS delay_days
    RETURN o.id AS order_id, s.name AS seller, s.state AS state,
           o.review_score AS review_score, delay_days,
           CASE WHEN delay_days > 0 THEN 'late' ELSE 'on_time' END AS status
    ORDER BY delay_days DESC
    LIMIT 20
    """,
    "review_sentiment": """
    MATCH (o:Order)-[:CONTAINS]->(p:Product)
    WHERE ($product IS NULL OR toLower(p.name) CONTAINS toLower($product))
      AND ($category IS NULL OR p.category CONTAINS $category)
    RETURN p.name AS product, p.category AS category, o.review_score AS review_score
    ORDER BY o.review_score DESC
    LIMIT 30
    """,
    "seller_performance": """
    MATCH (s:Seller)-[:SELLS]->(p:Product)
    OPTIONAL MATCH (o:Order)-[:SOLD_BY]->(s)
    WITH s, collect(distinct p.category) AS categories,
         avg(o.review_score) AS avg_score,
         avg(CASE WHEN o.delivery_date <= o.estimated_delivery_date THEN 1.0 ELSE 0 END) AS on_time_rate
    WHERE ($state IS NULL OR s.state = $state)
      AND ($min_reliability IS NULL OR s.reliability >= $min_reliability)
    RETURN s.name AS seller, s.state AS state, s.reliability AS reliability,
           avg_score, on_time_rate, categories
    ORDER BY reliability DESC, on_time_rate DESC
    LIMIT 15
    """,
    "state_trend": """
    MATCH (o:Order)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (o)-[:SOLD_BY]->(s:Seller)
    WHERE ($state IS NULL OR c.state = $state OR s.state = $state)
    WITH coalesce(c.state, s.state) AS state, count(o) AS orders,
         avg(o.review_score) AS avg_score
    RETURN state, orders, avg_score
    ORDER BY orders DESC
    LIMIT 20
    """,
    "category_insight": """
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(p)
    WHERE ($category IS NULL OR c.name CONTAINS $category)
    WITH c.name AS category, count(distinct p) AS products, count(o) AS orders,
         avg(o.review_score) AS avg_score, avg(p.price) AS avg_price
    RETURN category, products, orders, avg_score, avg_price
    ORDER BY orders DESC
    LIMIT 10
    """,
    "recommendation": """
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (p)<-[:SELLS]-(s:Seller)
    WHERE ($category IS NULL OR c.name CONTAINS $category)
      AND ($state IS NULL OR s.state = $state)
      AND ($min_rating IS NULL OR p.avg_rating >= $min_rating)
    RETURN p.id AS id, p.name AS name, p.category AS category, p.price AS price,
           p.avg_rating AS rating, s.name AS seller
    ORDER BY rating DESC, price ASC
    LIMIT 10
    """,
    "customer_behavior": """
    MATCH (c:Customer)-[:PLACED]->(o:Order)
    WITH c, count(o) AS orders, avg(o.review_score) AS avg_score
    RETURN c.id AS customer_id, orders, avg_score
    ORDER BY orders DESC
    LIMIT 15
    """,
    "delivery_impact_rule": """
    MATCH (o:Order)-[:CONTAINS]->(p:Product)
    WITH o, p,
         duration.inDays(o.estimated_delivery_date, o.delivery_date).days AS delay_days
    RETURN p.category AS category,
           avg(delay_days) AS avg_delay,
           avg(o.review_score) AS avg_score,
           corr(delay_days, o.review_score) AS delay_review_corr
    ORDER BY abs(delay_review_corr) DESC
    LIMIT 10
    """,
    "seller_reliability": """
    MATCH (s:Seller)-[:SELLS]->(:Product)
    OPTIONAL MATCH (o:Order)-[:SOLD_BY]->(s)
    WITH s,
         avg(CASE WHEN o.delivery_date <= o.estimated_delivery_date THEN 1.0 ELSE 0 END) AS on_time_rate,
         avg(o.review_score) AS avg_score
    WHERE ($state IS NULL OR s.state = $state)
    RETURN s.name AS seller, s.state AS state, s.reliability AS reliability,
           on_time_rate, avg_score
    ORDER BY on_time_rate DESC, avg_score DESC
    LIMIT 10
    """,
}


def build_query(intent: str, entities: EntityResult) -> Optional[Query]:
    template = QUERY_LIBRARY.get(intent)
    if not template:
        return None
    params = entities.to_params()
    return {"text": template, "params": params}
