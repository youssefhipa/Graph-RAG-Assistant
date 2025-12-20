import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(ROOT))

import pytest
from app.config import get_settings, EmbeddingModelConfig
from app.embedding import EmbeddingService


class TestEmbeddingModels:
    """Test embedding model configuration and selection."""
    
    def test_get_embedding_models_returns_dict(self):
        """Verify get_embedding_models returns correct structure."""
        settings = get_settings()
        models = settings.get_embedding_models()
        
        assert isinstance(models, dict)
        assert "model_1" in models
        assert isinstance(models["model_1"], EmbeddingModelConfig)
    
    def test_model_1_primary_exists(self):
        """Verify primary embedding model is configured."""
        settings = get_settings()
        models = settings.get_embedding_models()
        
        model1 = models["model_1"]
        assert model1.name == "Model 1 (Primary)"
        assert model1.model_id == settings.embed_model
        assert model1.vector_index == settings.vector_index
    
    def test_model_2_secondary_optional(self):
        """Verify secondary model is optional but present if configured."""
        settings = get_settings()
        models = settings.get_embedding_models()
        
        if settings.embed_model_2:
            assert "model_2" in models
            model2 = models["model_2"]
            assert model2.name == "Model 2 (Secondary)"
            assert model2.model_id == settings.embed_model_2
    
    def test_embedding_service_init_model_1(self):
        """Test EmbeddingService initialization with model_1."""
        settings = get_settings()
        service = EmbeddingService(settings, model_key="model_1")
        
        assert service.model_key == "model_1"
        assert service.model_config.name == "Model 1 (Primary)"
        assert service.model is not None
    
    def test_embedding_service_init_model_2_if_available(self):
        """Test EmbeddingService initialization with model_2 if available."""
        settings = get_settings()
        models = settings.get_embedding_models()
        
        if "model_2" in models:
            service = EmbeddingService(settings, model_key="model_2")
            assert service.model_key == "model_2"
            assert service.model_config.name == "Model 2 (Secondary)"
    
    def test_embedding_service_invalid_model_key(self):
        """Test EmbeddingService raises error for invalid model key."""
        settings = get_settings()
        
        with pytest.raises(ValueError, match="Model key 'invalid' not found"):
            EmbeddingService(settings, model_key="invalid")
    
    def test_embedding_service_default_model_key(self):
        """Test EmbeddingService defaults to model_1."""
        settings = get_settings()
        service = EmbeddingService(settings)  # No model_key specified
        
        assert service.model_key == "model_1"
    
    def test_embed_produces_vectors(self):
        """Test that embed() produces valid vectors."""
        settings = get_settings()
        service = EmbeddingService(settings, model_key="model_1")
        
        texts = ["test query", "another test"]
        vectors = service.embed(texts)
        
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(elem, float) for v in vectors for elem in v)
        assert all(len(v) > 0 for v in vectors)  # Non-empty vectors
    
    def test_embed_consistent_dimensions(self):
        """Test that embeddings have consistent dimensions."""
        settings = get_settings()
        service = EmbeddingService(settings, model_key="model_1")
        
        vectors = service.embed(["test 1", "test 2", "test 3"])
        dims = [len(v) for v in vectors]
        
        assert len(set(dims)) == 1, "All vectors should have same dimension"
    
    def test_different_models_have_different_configs(self):
        """Test that model_1 and model_2 have different configurations."""
        settings = get_settings()
        models = settings.get_embedding_models()
        
        if "model_2" in models:
            model1 = models["model_1"]
            model2 = models["model_2"]
            
            # Should have different model IDs or indices
            assert (model1.model_id != model2.model_id or 
                    model1.vector_index != model2.vector_index)