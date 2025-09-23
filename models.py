#!/usr/bin/env python3
"""Data models and schema definitions for Pet Food Query System"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class DatabaseSchema:
    """Database schema definition for the pet food table"""
    
    table_name: str = "pet_food"
    columns: List[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = [
                "id", "brand_name_display", "brand_name_english", "brand_name_kana",
                "empty_col", "product_name", "product_name_english", "product_name_kana",
                "variation_name", "variation_name_final", "variation_name_formula",
                "yumika_status", "final_confirmation_status", "nutrient_review_status",
                "target_animal_species", "food_genre", "food_type", "life_stage",
                "classification_by_activity_level", "specific_physical_condition",
                "therapeutic_food_category", "content_amount_label", "content_amount_g",
                "product_url", "ingredients", "type_of_meat", "legume_classification",
                "grain_classification", "type_of_fish", "additives_preservatives",
                "metabolizable_energy", "protein_label", "fat_label", "fiber_label",
                "ash_label", "moisture_label", "metabolizable_energy_100g",
                "protein_percent", "fat_percent", "carbohydrates_percent"
            ]
    
    def get_column_categories(self) -> Dict[str, List[str]]:
        """Get columns organized by category for better data formatting"""
        return {
            "basic_info": [
                'id', 'brand_name_display', 'brand_name_english', 'brand_name_kana',
                'empty_col', 'product_name', 'product_name_english', 'product_name_kana',
                'variation_name', 'variation_name_final', 'variation_name_formula',
                'target_animal_species', 'food_genre', 'food_type', 'life_stage'
            ],
            "nutritional": [
                'protein_percent', 'fat_percent', 'carbohydrates_percent',
                'metabolizable_energy', 'metabolizable_energy_100g',
                'protein_label', 'fat_label', 'fiber_label', 'ash_label', 'moisture_label'
            ],
            "ingredients": [
                'ingredients', 'type_of_meat', 'type_of_fish', 'legume_classification',
                'grain_classification', 'additives_preservatives'
            ],
            "status_and_other": [
                'yumika_status', 'final_confirmation_status', 'nutrient_review_status',
                'classification_by_activity_level', 'specific_physical_condition',
                'therapeutic_food_category', 'content_amount_label', 'content_amount_g',
                'product_url'
            ]
        }


@dataclass
class QueryResult:
    """Model for query results"""
    
    user_query: str
    sql_query: str
    results: Optional[Any]  # pandas DataFrame
    answer: str
    timestamp: str
    result_count: int = 0
    processing_time: float = 0.0
    error: bool = False
    error_message: Optional[str] = None
    saved_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_query": self.user_query,
            "sql_query": self.sql_query,
            "results": self.results,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "result_count": self.result_count,
            "processing_time": self.processing_time,
            "error": self.error,
            "error_message": self.error_message,
            "saved_file": self.saved_file
        }
    
    @classmethod
    def create_error_result(cls, user_query: str, sql_query: str, error_message: str, 
                           processing_time: float = 0.0) -> 'QueryResult':
        """Create an error result instance"""
        return cls(
            user_query=user_query,
            sql_query=sql_query,
            results=None,
            answer="Sorry, there was an error processing your query. Please try rephrasing your question.",
            timestamp=datetime.now().isoformat(),
            error=True,
            error_message=error_message,
            processing_time=processing_time
        )


@dataclass
class SQLValidationResult:
    """Model for SQL validation results"""
    
    is_valid: bool
    error_message: str = ""
    
    @classmethod
    def valid(cls) -> 'SQLValidationResult':
        """Create a valid result"""
        return cls(is_valid=True, error_message="Query is valid")
    
    @classmethod
    def invalid(cls, error_message: str) -> 'SQLValidationResult':
        """Create an invalid result with error message"""
        return cls(is_valid=False, error_message=error_message)


@dataclass
class DisplayColumns:
    """Column sets for different display modes"""
    
    comparison_view: List[str] = None
    raw_display: List[str] = None
    table_view: List[str] = None
    
    def __post_init__(self):
        if self.comparison_view is None:
            self.comparison_view = [
                'brand_name_english', 'product_name_english', 'target_animal_species',
                'food_type', 'protein_percent', 'fat_percent', 'carbohydrates_percent'
            ]
        
        if self.raw_display is None:
            self.raw_display = [
                'id', 'brand_name_english', 'product_name_english', 'variation_name_final',
                'target_animal_species', 'food_type', 'life_stage', 'protein_percent',
                'fat_percent', 'carbohydrates_percent', 'metabolizable_energy_100g'
            ]
        
        if self.table_view is None:
            self.table_view = [
                'brand_name_english', 'product_name_english', 'target_animal_species',
                'food_type', 'protein_percent', 'fat_percent', 'carbohydrates_percent'
            ]


# Default schema instance
DEFAULT_SCHEMA = DatabaseSchema()

# Default display columns instance
DEFAULT_DISPLAY_COLUMNS = DisplayColumns()
