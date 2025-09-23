#!/usr/bin/env python3
"""Configuration management for Pet Food Query System"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the Pet Food Query System"""
    
    # Database Configuration
    DATABASE_URL = os.getenv(
        'DATABASE_URL', 
        'postgresql://pguser:pguser@localhost:5454/nutrient_food'
    )
    
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    # Application Configuration
    RESULTS_FOLDER = Path("results")
    MAX_RESULTS_LIMIT = 15
    MAX_DISPLAY_PRODUCTS = 8
    MAX_RAW_DISPLAY_PRODUCTS = 10
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Query Configuration
    QUERY_TIMEOUT = int(os.getenv('QUERY_TIMEOUT', '30'))  # seconds
    
    @classmethod
    def validate_config(cls) -> tuple[bool, str]:
        """
        Validate that all required configuration is present
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not cls.GEMINI_API_KEY:
            return False, "GEMINI_API_KEY environment variable is required"
        
        if not cls.DATABASE_URL:
            return False, "DATABASE_URL is required"
        
        return True, "Configuration is valid"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.RESULTS_FOLDER.mkdir(exist_ok=True)


# Common query patterns and examples
QUERY_EXAMPLES = {
    "nutritional_queries": [
        "high protein dog food",
        "low fat cat food", 
        "food with more than 30% protein"
    ],
    "ingredient_queries": [
        "chicken-based dog food",
        "grain-free cat food",
        "food with salmon"
    ],
    "life_stage_queries": [
        "puppy food",
        "senior cat food",
        "kitten food"
    ],
    "brand_queries": [
        "AATU brand products",
        "show me all Royal Canin products"
    ]
}

# SQL Security Configuration
DANGEROUS_SQL_KEYWORDS = [
    'drop', 'delete', 'update', 'insert', 'alter', 'create',
    'truncate', 'grant', 'revoke', 'exec', 'execute'
]
