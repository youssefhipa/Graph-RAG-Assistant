from __future__ import annotations

from typing import Any, Dict, Optional

from .entities import EntityResult


Query = Dict[str, Any]


QUERY_LIBRARY: Dict[str, str] = {
    "seller_count": """
    MATCH (oi:OrderItem)
    WITH coalesce(oi.seller_id, oi.sellerId, oi.seller) AS seller_id
    WHERE seller_id IS NOT NULL
    RETURN count(distinct seller_id) AS seller_count
    """,
    "product_search": """
    MATCH (p:Product)
    OPTIONAL MATCH (oi:OrderItem)-[:REFERS_TO]->(p)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (o)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WITH p, c,
         avg(coalesce(r.review_score, o.review_score)) AS rating
    WHERE ($category IS NULL OR p.product_category_name CONTAINS $category OR p.category CONTAINS $category)
      AND ($state IS NULL OR c.customer_state = $state)
      AND ($city IS NULL OR c.customer_city IS NOT NULL AND toLower(c.customer_city) CONTAINS toLower($city))
      AND ($min_rating IS NULL OR rating IS NOT NULL AND rating >= $min_rating)
    RETURN p.product_id AS id, coalesce(p.name, p.product_id) AS name, p.product_category_name AS category,
           p.price AS price, rating AS rating, c.customer_state AS customer_state, c.customer_city AS customer_city
    ORDER BY (rating IS NULL) ASC, rating DESC, price ASC
    LIMIT 15
    """,
    "delivery_delay": """
    MATCH (o:Order)<-[:PLACED]-(c:Customer)
    WHERE ($state IS NULL OR c.customer_state = $state)
      AND ($start_date IS NULL OR coalesce(o.purchase_date, o.order_purchase_timestamp) IS NOT NULL
           AND date(coalesce(o.purchase_date, o.order_purchase_timestamp)) >= date($start_date))
      AND ($end_date IS NULL OR coalesce(o.purchase_date, o.order_purchase_timestamp) IS NOT NULL
           AND date(coalesce(o.purchase_date, o.order_purchase_timestamp)) <= date($end_date))
    WITH o, c,
         coalesce(
            o.delivery_delay_days,
            CASE
                WHEN coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date) IS NOT NULL
                 AND coalesce(o.delivery_date, o.order_delivered_customer_date) IS NOT NULL
                THEN duration.inDays(
                    date(coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date)),
                    date(coalesce(o.delivery_date, o.order_delivered_customer_date))
                ).days
                ELSE 0
            END
         ) AS delay_days,
         coalesce(o.review_score, o.reviewScore) AS review_score
    RETURN o.id AS order_id, c.customer_state AS state,
           review_score AS review_score, delay_days,
           CASE WHEN delay_days > 0 THEN 'late' ELSE 'on_time' END AS status
    ORDER BY delay_days DESC
    LIMIT 20
    """,
    "review_sentiment": """
    MATCH (p:Product)<-[:REFERS_TO]-(oi:OrderItem)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WHERE ($product IS NULL OR p.name IS NOT NULL AND toLower(p.name) CONTAINS toLower($product))
      AND (
        $category IS NULL
        OR p.product_category_name IS NOT NULL AND p.product_category_name CONTAINS $category
        OR p.category IS NOT NULL AND p.category CONTAINS $category
      )
    WITH p, coalesce(r.review_score, o.review_score) AS review_score
    RETURN p.name AS product, p.product_category_name AS category, review_score
    ORDER BY review_score DESC
    LIMIT 30
    """,
    "seller_performance": """
    MATCH (oi:OrderItem)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (o)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WITH coalesce(oi.seller_id, oi.sellerId, oi.seller) AS seller_id,
         collect(distinct oi.product_id) AS products,
         avg(coalesce(r.review_score, o.review_score, 0)) AS avg_score,
         avg(
            CASE WHEN coalesce(o.delivery_date, o.order_delivered_customer_date) IS NOT NULL
                   AND coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date) IS NOT NULL
                   AND date(coalesce(o.delivery_date, o.order_delivered_customer_date))
                       <= date(coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date))
                 THEN 1.0 ELSE 0 END
         ) AS on_time_rate,
         collect(distinct c.customer_state) AS states
    WHERE seller_id IS NOT NULL
      AND ($state IS NULL OR $state IN states)
      AND ($min_reliability IS NULL OR on_time_rate >= $min_reliability)
    RETURN seller_id AS seller, avg_score, on_time_rate, products
    ORDER BY on_time_rate DESC, avg_score DESC
    LIMIT 15
    """,
    "state_trend": """
    MATCH (o:Order)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WHERE ($state IS NULL OR c.customer_state = $state)
    WITH c.customer_state AS state, count(o) AS orders,
         avg(coalesce(r.review_score, o.review_score, 0)) AS avg_score
    WHERE state IS NOT NULL
    RETURN state, orders, avg_score
    ORDER BY orders DESC
    LIMIT 20
    """,
    "category_insight": """
    MATCH (p:Product)
    OPTIONAL MATCH (oi:OrderItem)-[:REFERS_TO]->(p)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WHERE ($category IS NULL OR p.product_category_name CONTAINS $category OR p.category CONTAINS $category)
    WITH p.product_category_name AS category, count(distinct p) AS products, count(o) AS orders,
         avg(coalesce(r.review_score, o.review_score, 0)) AS avg_score,
         avg(coalesce(oi.price, p.price)) AS avg_price
    RETURN category, products, orders, avg_score, avg_price
    ORDER BY orders DESC
    LIMIT 10
    """,
    "recommendation": """
    MATCH (p:Product)
    OPTIONAL MATCH (oi:OrderItem)-[:REFERS_TO]->(p)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (o)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WITH p, c, avg(coalesce(r.review_score, o.review_score)) AS rating
    WHERE ($category IS NULL OR p.product_category_name CONTAINS $category OR p.category CONTAINS $category)
      AND ($state IS NULL OR c.customer_state = $state)
      AND ($min_rating IS NULL OR rating IS NOT NULL AND rating >= $min_rating)
    RETURN p.product_id AS id, coalesce(p.name, p.product_id) AS name,
           p.product_category_name AS category, p.price AS price, rating AS rating
    ORDER BY rating DESC, price ASC
    LIMIT 10
    """,
    "customer_behavior": """
    MATCH (c:Customer)-[:PLACED]->(o:Order)
    WITH c, count(o) AS orders, 
         avg(CASE WHEN o.review_score IS NOT NULL THEN o.review_score ELSE 0 END) AS avg_score
    RETURN c.id AS customer_id, orders, avg_score
    ORDER BY orders DESC
    LIMIT 15
    """,
    "delivery_impact_rule": """
    MATCH (p:Product)<-[:REFERS_TO]-(oi:OrderItem)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WHERE p.product_category_name IS NOT NULL
      AND coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date) IS NOT NULL
      AND coalesce(o.delivery_date, o.order_delivered_customer_date) IS NOT NULL
      AND coalesce(r.review_score, o.review_score) IS NOT NULL
    WITH p.product_category_name AS category,
         duration.inDays(
            date(coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date)),
            date(coalesce(o.delivery_date, o.order_delivered_customer_date))
         ).days AS delay_days,
         coalesce(r.review_score, o.review_score) AS review_score
    WITH category,
         avg(delay_days) AS avg_delay,
         avg(review_score) AS avg_score,
         corr(delay_days, review_score) AS delay_review_corr
    WHERE category IS NOT NULL
    RETURN category,
           avg_delay,
           avg_score,
           CASE WHEN delay_review_corr IS NaN THEN 0 ELSE delay_review_corr END AS delay_review_corr
    ORDER BY abs(CASE WHEN delay_review_corr IS NaN THEN 0 ELSE delay_review_corr END) DESC
    LIMIT 10
    """,
    "seller_reliability": """
    MATCH (oi:OrderItem)
    OPTIONAL MATCH (o:Order)-[:CONTAINS]->(oi)
    OPTIONAL MATCH (o)<-[:PLACED]-(c:Customer)
    OPTIONAL MATCH (r:Review)-[:REFERS_TO]->(o)
    WITH coalesce(oi.seller_id, oi.sellerId, oi.seller) AS seller_id,
         avg(
            CASE WHEN coalesce(o.delivery_date, o.order_delivered_customer_date) IS NOT NULL
                   AND coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date) IS NOT NULL
                   AND date(coalesce(o.delivery_date, o.order_delivered_customer_date))
                       <= date(coalesce(o.estimated_delivery_date, o.order_estimated_delivery_date))
                 THEN 1.0 ELSE 0 END
         ) AS on_time_rate,
         avg(coalesce(r.review_score, o.review_score, 0)) AS avg_score,
         collect(distinct c.customer_state) AS states
    WHERE seller_id IS NOT NULL
      AND ($state IS NULL OR $state IN states)
    RETURN seller_id AS seller, on_time_rate, avg_score
    ORDER BY on_time_rate DESC, avg_score DESC
    LIMIT 10
    """,
}


def build_query(intent: str, entities: EntityResult) -> Optional[Query]:
    """
    Build a Cypher query from an intent and extracted entities.
    
    Args:
        intent: The classified intent (should match a key in QUERY_LIBRARY).
        entities: Extracted entities from the user's question.
        
    Returns:
        A dict with 'text' (Cypher) and 'params' (parameter dict), or None if intent not found.
    """
    template = QUERY_LIBRARY.get(intent)
    if not template:
        return None
    params = entities.to_params()
    return {"text": template, "params": params}


def validate_query_template(intent: str) -> bool:
    """
    Validate that a query template exists and is executable.
    
    Args:
        intent: The intent to validate.
        
    Returns:
        True if the template exists, False otherwise.
    """
    return intent in QUERY_LIBRARY


def list_all_templates() -> Dict[str, str]:
    """
    Get all available query templates.
    
    Returns:
        Dictionary mapping intent names to their Cypher templates.
    """
    return QUERY_LIBRARY.copy()
