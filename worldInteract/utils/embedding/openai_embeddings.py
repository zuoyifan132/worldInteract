"""
OpenAI Embeddings API wrapper for tool dependency analysis.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from worldInteract.utils.config_manager import config_manager


class OpenAIEmbeddings:
    """OpenAI embeddings client for vectorizing tool descriptions."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize OpenAI embeddings client.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.embedding_config = self.config_manager.get_model_config("embedding")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.embedding_config.get("api_base", "https://api.openai.com/v1")
        )
        
        self.model = self.embedding_config.get("model", "text-embedding-3-large")
        self.dimensions = self.embedding_config.get("dimensions", 3072)
        self.batch_size = self.embedding_config.get("batch_size", 100)
        
        logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts in batches to handle rate limits.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
        
        return all_embeddings
    
    def embed_tool_parameters(self, tool: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate embeddings for all parameters of a tool.
        
        Args:
            tool: Tool dictionary with 'parameters' field
            
        Returns:
            Dictionary mapping parameter names to their embeddings
        """
        parameters = tool.get("parameters", {})
        if not parameters:
            return {}
        
        param_descriptions = []
        param_names = []
        
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict) and "description" in param_info:
                param_descriptions.append(param_info["description"])
                param_names.append(param_name)
        
        if not param_descriptions:
            return {}
        
        embeddings = self.embed_texts(param_descriptions)
        
        return {
            param_names[i]: embeddings[i] 
            for i in range(len(param_names))
        }
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def calculate_tool_similarity(
        self, 
        tool1_embeddings: Dict[str, List[float]], 
        tool2_embeddings: Dict[str, List[float]]
    ) -> float:
        """
        Calculate maximum similarity between parameters of two tools.
        
        Args:
            tool1_embeddings: Parameter embeddings for tool 1
            tool2_embeddings: Parameter embeddings for tool 2
            
        Returns:
            Maximum cosine similarity between any pair of parameters
        """
        if not tool1_embeddings or not tool2_embeddings:
            return 0.0
        
        max_similarity = 0.0
        
        for param1_name, embedding1 in tool1_embeddings.items():
            for param2_name, embedding2 in tool2_embeddings.items():
                similarity = self.cosine_similarity(embedding1, embedding2)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
