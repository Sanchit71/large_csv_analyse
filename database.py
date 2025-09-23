#!/usr/bin/env python3
"""Database operations and SQL validation for Pet Food Query System"""

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
import logging

from config import Config, DANGEROUS_SQL_KEYWORDS
from models import DatabaseSchema, SQLValidationResult, DEFAULT_SCHEMA

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: str, schema: DatabaseSchema = None):
        """
        Initialize the Database Manager
        
        Args:
            database_url: PostgreSQL connection string
            schema: Database schema definition
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.schema = schema or DEFAULT_SCHEMA
    
    def validate_sql_query(self, sql_query: str) -> SQLValidationResult:
        """
        Validate SQL query for safety and correctness
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            SQLValidationResult with validation status and error message
        """
        # Remove whitespace and convert to lowercase for checking
        query_lower = sql_query.lower().strip()
        
        # Security checks - block dangerous operations
        for keyword in DANGEROUS_SQL_KEYWORDS:
            if keyword in query_lower:
                return SQLValidationResult.invalid(f"Dangerous SQL keyword '{keyword}' detected")
        
        # Must be a SELECT query
        if not query_lower.startswith('select'):
            return SQLValidationResult.invalid("Only SELECT queries are allowed")
        
        # Check for valid table name
        if self.schema.table_name not in query_lower:
            return SQLValidationResult.invalid(f"Query must reference table '{self.schema.table_name}'")
        
        # Basic syntax validation
        if query_lower.count('(') != query_lower.count(')'):
            return SQLValidationResult.invalid("Mismatched parentheses in query")
        
        return SQLValidationResult.valid()
    
    def execute_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query against the database
        
        Args:
            sql_query: SQL query string
            
        Returns:
            DataFrame with query results or None if error
        """
        try:
            logger.info(f"Executing query: {sql_query}")
            df = pd.read_sql(text(sql_query), self.engine)
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            # Try a simplified fallback query
            if "WHERE" in sql_query.upper():
                try:
                    simple_query = f"SELECT * FROM {self.schema.table_name} LIMIT {Config.MAX_RESULTS_LIMIT}"
                    logger.info("Trying fallback query")
                    df = pd.read_sql(text(simple_query), self.engine)
                    return df
                except Exception:
                    pass
            return None
    
    def get_fallback_query(self, user_query: str) -> str:
        """
        Generate fallback query based on common patterns - ALWAYS uses SELECT *
        
        Args:
            user_query: Original user query for pattern matching
            
        Returns:
            Fallback SQL query string
        """
        query_lower = user_query.lower()
        
        # Basic pattern matching for fallback - all use SELECT * to get ALL columns
        if any(word in query_lower for word in ['dog', 'canine']):
            return f"SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%dog%' LIMIT {Config.MAX_RESULTS_LIMIT}"
        elif any(word in query_lower for word in ['cat', 'feline', 'kitten']):
            return f"SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%cat%' LIMIT {Config.MAX_RESULTS_LIMIT}"
        elif any(word in query_lower for word in ['protein', 'high protein']):
            return f"SELECT * FROM {self.schema.table_name} WHERE protein_percent ILIKE '%high%' OR ingredients ILIKE '%protein%' LIMIT {Config.MAX_RESULTS_LIMIT}"
        else:
            return f"SELECT * FROM {self.schema.table_name} ORDER BY id DESC LIMIT {Config.MAX_RESULTS_LIMIT}"
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test database connection
        
        Returns:
            Tuple of (is_connected, error_message)
        """
        try:
            # Try a simple query to test connection
            test_query = f"SELECT COUNT(*) FROM {self.schema.table_name} LIMIT 1"
            result = pd.read_sql(text(test_query), self.engine)
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"


class SQLQueryBuilder:
    """Helper class for building SQL queries"""
    
    def __init__(self, schema: DatabaseSchema = None):
        """
        Initialize SQL Query Builder
        
        Args:
            schema: Database schema definition
        """
        self.schema = schema or DEFAULT_SCHEMA
    
    def get_schema_info(self) -> str:
        """Get formatted schema information for AI prompts"""
        return f"""
        Table: {self.schema.table_name}
        Columns: {', '.join(self.schema.columns)}
        
        Important column descriptions:
        - id: Primary key (INTEGER - only numeric column)
        - target_animal_species: Type of animal (dog, cat, etc.) - TEXT
        - food_genre: Genre of food (dry food, wet food, treats, etc.) - TEXT
        - food_type: Specific type of food - TEXT
        - life_stage: Age group (puppy, adult, senior, etc.) - TEXT
        - specific_physical_condition: Health conditions - TEXT
        - therapeutic_food_category: Medical/therapeutic purposes - TEXT
        - ingredients: List of ingredients - TEXT
        - protein_percent, fat_percent, carbohydrates_percent: Nutritional content - TEXT (need CAST)
        - metabolizable_energy, content_amount_g: Energy/weight content - TEXT (need CAST)
        - brand_name_english: English brand name - TEXT
        - product_name_english: English product name - TEXT
        
        CRITICAL NOTES:
        - ALL columns except 'id' are stored as TEXT
        - For numeric comparisons, ALWAYS use: CAST(column_name AS FLOAT) or CAST(column_name AS INTEGER)
        - Use ILIKE for case-insensitive text searches
        - Handle NULL and empty string values properly
        - Some numeric columns may contain non-numeric text like "high", "low", etc.
        """
    
    def get_query_patterns(self) -> str:
        """Get SQL query pattern examples for AI prompts"""
        return f"""
        ADVANCED PATTERN EXAMPLES (ALWAYS USE SELECT *):
        
        Animal Type Queries:
        - "dog food" → SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%dog%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "cat treats" → SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%cat%' AND food_genre ILIKE '%treat%' LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Life Stage Queries:
        - "puppy food" → SELECT * FROM {self.schema.table_name} WHERE life_stage ILIKE '%puppy%' OR life_stage ILIKE '%young%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "senior cat" → SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%cat%' AND life_stage ILIKE '%senior%' LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Nutritional Queries (with safe numeric handling):
        - "high protein" → SELECT * FROM {self.schema.table_name} WHERE (protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "low fat" → SELECT * FROM {self.schema.table_name} WHERE (fat_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(fat_percent, '%', '') AS FLOAT) < 10) OR fat_percent ILIKE '%low%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "more than 30% protein" → SELECT * FROM {self.schema.table_name} WHERE protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 30 LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Ingredient Queries:
        - "chicken" → SELECT * FROM {self.schema.table_name} WHERE ingredients ILIKE '%chicken%' OR type_of_meat ILIKE '%chicken%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "grain free" → SELECT * FROM {self.schema.table_name} WHERE grain_classification ILIKE '%none%' OR grain_classification ILIKE '%grain%free%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "salmon" → SELECT * FROM {self.schema.table_name} WHERE ingredients ILIKE '%salmon%' OR type_of_fish ILIKE '%salmon%' LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Brand Queries:
        - "AATU" → SELECT * FROM {self.schema.table_name} WHERE brand_name_english ILIKE '%AATU%' OR brand_name_display ILIKE '%AATU%' LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Food Type Queries:
        - "dry food" → SELECT * FROM {self.schema.table_name} WHERE food_type ILIKE '%dry%' LIMIT {Config.MAX_RESULTS_LIMIT}
        - "wet food" → SELECT * FROM {self.schema.table_name} WHERE food_type ILIKE '%wet%' OR food_type ILIKE '%can%' LIMIT {Config.MAX_RESULTS_LIMIT}
        
        Combined Queries:
        - "high protein dog dry food" → SELECT * FROM {self.schema.table_name} WHERE target_animal_species ILIKE '%dog%' AND food_type ILIKE '%dry%' AND ((protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%') LIMIT {Config.MAX_RESULTS_LIMIT}
        
        CRITICAL NUMERIC HANDLING:
        - Use REPLACE(column, '%', '') to remove % signs before CAST
        - Use regex E'^[0-9]+\\.?[0-9]*%?$' to validate numeric values
        - Always provide fallback with ILIKE for text matches
        - Combine numeric and text searches with OR
        
        MANDATORY FORMAT:
        Your response must ALWAYS start with "SELECT * FROM {self.schema.table_name} WHERE..." and end with "LIMIT {Config.MAX_RESULTS_LIMIT}"
        NEVER use specific column names in SELECT clause - ALWAYS use SELECT *
        """
