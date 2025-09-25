"""
Similarity method enumeration for API deduplication.
"""

from enum import Enum


class SimilarityMethod(Enum):
    """Enumeration class defining similarity calculation methods for API deduplication"""
    
    EMBEDDING_MODEL = "embedding_model"
    """Use embedding model for semantic similarity calculation (more accurate but requires API key and slower)"""
    
    SEQUENCE_MATCHER = "sequence_matcher"
    """Use SequenceMatcher for text similarity calculation (faster, no external dependencies)"""
    
    @classmethod
    def from_string(cls, value: str) -> 'SimilarityMethod':
        """
        Create SimilarityMethod enum instance from string
        
        Args:
            value: String value
            
        Returns:
            SimilarityMethod enum instance
            
        Raises:
            ValueError: If the provided string is not a valid similarity method
        """
        for method in cls:
            if method.value == value:
                return method
        
        valid_values = [method.value for method in cls]
        raise ValueError(f"Invalid similarity method '{value}'. Valid options: {valid_values}")
    
    @classmethod
    def get_default(cls) -> 'SimilarityMethod':
        """Get the default similarity calculation method"""
        return cls.EMBEDDING_MODEL
    
    @classmethod
    def get_all_values(cls) -> list[str]:
        """Get all valid string values"""
        return [method.value for method in cls]
    
    def __str__(self) -> str:
        """Return the string value of the enum"""
        return self.value
    
    def is_embedding_based(self) -> bool:
        """Check if this is an embedding-based method"""
        return self == self.EMBEDDING_MODEL
    
    def is_text_based(self) -> bool:
        """Check if this is a text-based method"""
        return self == self.SEQUENCE_MATCHER
