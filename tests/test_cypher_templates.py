import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(ROOT))

import pytest
from app.queries import (
    build_query, 
    validate_query_template, 
    list_all_templates,
    QUERY_LIBRARY
)
from app.entities import EntityResult


class TestCypherTemplates:
    """Test that all 10+ Cypher templates handle NULL parameters."""
    
    def test_all_templates_exist(self):
        """Verify all 11 templates exist in library."""
        assert len(QUERY_LIBRARY) >= 10
        expected_intents = {
            "seller_count",
            "product_search",
            "delivery_delay",
            "review_sentiment",
            "seller_performance",
            "state_trend",
            "category_insight",
            "recommendation",
            "customer_behavior",
            "delivery_impact_rule",
            "seller_reliability",
        }
        assert all(intent in QUERY_LIBRARY for intent in expected_intents)
    
    def test_validate_query_template_existing(self):
        """Test validate_query_template returns True for existing templates."""
        assert validate_query_template("seller_count") is True
        assert validate_query_template("product_search") is True
        assert validate_query_template("delivery_delay") is True
    
    def test_validate_query_template_non_existing(self):
        """Test validate_query_template returns False for non-existing templates."""
        assert validate_query_template("invalid_intent") is False
        assert validate_query_template("unknown") is False
    
    def test_list_all_templates_returns_dict(self):
        """Test list_all_templates returns proper dictionary."""
        templates = list_all_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) >= 10
        assert all(isinstance(k, str) and isinstance(v, str) 
                   for k, v in templates.items())
    
    def test_build_query_with_empty_entities(self):
        """Test that queries work with ALL NULL parameters."""
        empty_entities = EntityResult()
        
        # Test all intents with empty params
        for intent in QUERY_LIBRARY.keys():
            query = build_query(intent, empty_entities)
            
            assert query is not None, f"Query should exist for {intent}"
            assert "text" in query
            assert "params" in query
            assert isinstance(query["text"], str)
            assert isinstance(query["params"], dict)
            
            # Verify no required parameters
            assert query["params"] == {}
    
    def test_product_search_null_params(self):
        """Test product_search template with NULL parameters."""
        entities = EntityResult(
            category=None,
            state=None,
            city=None,
            min_rating=None
        )
        query = build_query("product_search", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$category IS NULL" in cypher
        assert "$state IS NULL" in cypher
        assert "$city IS NULL" in cypher
        assert "$min_rating IS NULL" in cypher
    
    def test_delivery_delay_null_params(self):
        """Test delivery_delay template with NULL parameters."""
        entities = EntityResult(
            state=None,
            start_date=None,
            end_date=None
        )
        query = build_query("delivery_delay", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$state IS NULL" in cypher
        assert "$start_date IS NULL" in cypher
        assert "$end_date IS NULL" in cypher
        
        # Verify NULL checks for date comparisons
        assert "IS NOT NULL" in cypher
        assert "CASE WHEN" in cypher  # For handling nulls in calculations
    
    def test_seller_performance_null_params(self):
        """Test seller_performance template with NULL parameters."""
        entities = EntityResult(
            state=None,
            min_reliability=None
        )
        query = build_query("seller_performance", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$state IS NULL" in cypher
        assert "$min_reliability IS NULL" in cypher
        assert "IS NOT NULL" in cypher  # For null safety in aggregations
    
    def test_state_trend_null_params(self):
        """Test state_trend template with NULL parameters."""
        entities = EntityResult(state=None)
        query = build_query("state_trend", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$state IS NULL" in cypher
        assert "CASE WHEN" in cypher  # For safe aggregations
    
    def test_category_insight_null_params(self):
        """Test category_insight template with NULL parameters."""
        entities = EntityResult(category=None)
        query = build_query("category_insight", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$category IS NULL" in cypher
        assert "CASE WHEN" in cypher  # For safe aggregations
    
    def test_review_sentiment_null_params(self):
        """Test review_sentiment template with NULL parameters."""
        entities = EntityResult(
            product=None,
            category=None
        )
        query = build_query("review_sentiment", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$product IS NULL" in cypher
        assert "$category IS NULL" in cypher
        assert "IS NOT NULL" in cypher  # For string operations
    
    def test_delivery_impact_rule_null_params(self):
        """Test delivery_impact_rule template with NULL parameters."""
        entities = EntityResult()
        query = build_query("delivery_impact_rule", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks for required fields
        assert "IS NOT NULL" in cypher
        # Verify NaN handling
        assert "IS NaN" in cypher
    
    def test_seller_reliability_null_params(self):
        """Test seller_reliability template with NULL parameters."""
        entities = EntityResult(state=None)
        query = build_query("seller_reliability", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$state IS NULL" in cypher
        assert "IS NOT NULL" in cypher  # For date comparisons
    
    def test_customer_behavior_null_params(self):
        """Test customer_behavior template with NULL parameters."""
        entities = EntityResult()
        query = build_query("customer_behavior", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify safe aggregations
        assert "CASE WHEN" in cypher
    
    def test_recommendation_null_params(self):
        """Test recommendation template with NULL parameters."""
        entities = EntityResult(
            category=None,
            state=None,
            min_rating=None
        )
        query = build_query("recommendation", entities)
        
        assert query is not None
        cypher = query["text"]
        
        # Verify NULL checks exist
        assert "$category IS NULL" in cypher
        assert "$state IS NULL" in cypher
        assert "$min_rating IS NULL" in cypher
    
    def test_seller_count_null_params(self):
        """Test seller_count template (simplest query)."""
        entities = EntityResult()
        query = build_query("seller_count", entities)
        
        assert query is not None
        assert query["params"] == {}
    
    def test_build_query_invalid_intent(self):
        """Test build_query returns None for invalid intent."""
        entities = EntityResult()
        query = build_query("invalid_intent", entities)
        
        assert query is None
    
    def test_all_templates_have_proper_structure(self):
        """Verify all templates have proper Cypher syntax."""
        for intent, cypher in QUERY_LIBRARY.items():
            assert isinstance(cypher, str)
            assert len(cypher.strip()) > 0
            # Basic Cypher validation
            assert "MATCH" in cypher or "CALL" in cypher
            assert "RETURN" in cypher