import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(ROOT))

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.kg_client import KGClient
from app.config import get_settings


class TestVectorQueries:
    """Test Neo4j vector query validation."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.neo4j_uri = "neo4j://localhost:7687"
        settings.neo4j_user = "neo4j"
        settings.neo4j_password = "password"
        settings.neo4j_database = "neo4j"
        settings.vector_index = "test_index"
        settings.embed_property = "embedding"
        return settings
    
    def test_vector_query_empty_vector_raises_error(self, mock_settings):
        """Test that empty vector raises ValueError."""
        with patch('app.kg_client.GraphDatabase'):
            client = KGClient(mock_settings)
            
            with pytest.raises(ValueError, match="Vector cannot be empty"):
                client.vector_query(vector=[])
    
    def test_vector_query_invalid_top_k_raises_error(self, mock_settings):
        """Test that invalid top_k raises ValueError."""
        with patch('app.kg_client.GraphDatabase'):
            client = KGClient(mock_settings)
            
            with pytest.raises(ValueError, match="top_k must be at least 1"):
                client.vector_query(vector=[0.1, 0.2, 0.3], top_k=0)
            
            with pytest.raises(ValueError, match="top_k must be at least 1"):
                client.vector_query(vector=[0.1, 0.2, 0.3], top_k=-1)
    
    def test_vector_query_invalid_index_name_raises_error(self, mock_settings):
        """Test that invalid index name raises ValueError."""
        with patch('app.kg_client.GraphDatabase'):
            # Test with settings that has no default index
            mock_settings_no_index = Mock()
            mock_settings_no_index.neo4j_uri = "neo4j://localhost:7687"
            mock_settings_no_index.neo4j_user = "neo4j"
            mock_settings_no_index.neo4j_password = "password"
            mock_settings_no_index.neo4j_database = "neo4j"
            mock_settings_no_index.vector_index = None  # Invalid default!
            mock_settings_no_index.embed_property = "embedding"
            
            client_no_index = KGClient(mock_settings_no_index)
            
            # Should raise error when both index_name and settings.vector_index are None
            with pytest.raises(ValueError, match="Invalid index name"):
                client_no_index.vector_query(
                    vector=[0.1, 0.2, 0.3],
                    index_name=None
                )
            
            # Also should raise when explicitly passing None and default is None
            with pytest.raises(ValueError, match="Invalid index name"):
                client_no_index.vector_query(
                    vector=[0.1, 0.2, 0.3],
                    index_name=""  # Empty string falls back to None
                )
    
    def test_vector_query_default_index_name(self, mock_settings):
        """Test that default index name is used."""
        with patch('app.kg_client.GraphDatabase') as mock_db:
            mock_session = MagicMock()
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = []
            
            client = KGClient(mock_settings)
            client.vector_query(vector=[0.1, 0.2, 0.3])
            
            # Verify the cypher query was called
            assert mock_session.run.called
    
    def test_vector_query_custom_index_name(self, mock_settings):
        """Test that custom index name is respected."""
        with patch('app.kg_client.GraphDatabase') as mock_db:
            mock_session = MagicMock()
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = []
            
            client = KGClient(mock_settings)
            client.vector_query(
                vector=[0.1, 0.2, 0.3],
                top_k=5,
                index_name="custom_index"
            )
            
            # Verify cypher query contains custom index name
            call_args = mock_session.run.call_args
            cypher_query = call_args[0][0]
            assert "custom_index" in cypher_query
    
    def test_vector_query_returns_list(self, mock_settings):
        """Test that vector_query returns a list."""
        with patch('app.kg_client.GraphDatabase') as mock_db:
            mock_session = MagicMock()
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            
            mock_record = MagicMock()
            mock_record.data.return_value = {"item": {"id": "1"}, "score": 0.95}
            mock_session.run.return_value = [mock_record]
            
            client = KGClient(mock_settings)
            result = client.vector_query(vector=[0.1, 0.2, 0.3])
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert "score" in result[0]
    
    def test_vector_query_handles_no_results(self, mock_settings):
        """Test that vector_query handles empty results gracefully."""
        with patch('app.kg_client.GraphDatabase') as mock_db:
            mock_session = MagicMock()
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = []
            
            client = KGClient(mock_settings)
            result = client.vector_query(vector=[0.1, 0.2, 0.3])
            
            assert isinstance(result, list)
            assert len(result) == 0
    
    def test_vector_query_handles_errors(self, mock_settings):
        """Test that vector_query handles exceptions properly."""
        with patch('app.kg_client.GraphDatabase') as mock_db:
            mock_session = MagicMock()
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.side_effect = Exception("Index not found")
            
            client = KGClient(mock_settings)
            
            with pytest.raises(RuntimeError, match="Vector query failed"):
                client.vector_query(vector=[0.1, 0.2, 0.3])