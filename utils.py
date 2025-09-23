#!/usr/bin/env python3
"""Utility functions and helpers for Pet Food Query System"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

from config import Config

# Set up logging
logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for the Pet Food Query System"""
    
    def __init__(self, results_folder: Path = None):
        """
        Initialize File Manager
        
        Args:
            results_folder: Path to results folder (defaults to Config.RESULTS_FOLDER)
        """
        self.results_folder = results_folder or Config.RESULTS_FOLDER
        self.results_folder.mkdir(exist_ok=True)
    
    def save_result_to_file(self, result: Dict[str, Any]) -> str:
        """
        Save query result to a text file in the results folder
        
        Args:
            result: Dictionary containing query result data
            
        Returns:
            Path to the saved file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean user query for filename (remove special characters)
        clean_query = "".join(c for c in result['user_query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_query = clean_query.replace(' ', '_')[:50]  # Limit length
        filename = f"query_{timestamp}_{clean_query}.txt"
        file_path = self.results_folder / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("PET FOOD QUERY SYSTEM - RESULT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Timestamp: {result.get('timestamp', 'N/A')}\n")
                f.write(f"User Query: {result['user_query']}\n")
                f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
                f.write(f"Results Found: {result.get('result_count', 0)} products\n\n")
                
                f.write("SQL Query:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{result['sql_query']}\n\n")
                
                f.write("AI Generated Answer:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{result['answer']}\n\n")
                
                if result.get('error'):
                    f.write("Error Information:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Error: {result.get('error_message', 'Unknown error')}\n\n")
                
                f.write("="*80 + "\n")
                f.write("End of Result\n")
                f.write("="*80 + "\n")
            
            logger.info(f"Result saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")
            return ""
    
    def save_dataframe_to_csv(self, df, filename_prefix: str = "query_results") -> str:
        """
        Save DataFrame to CSV file
        
        Args:
            df: pandas DataFrame to save
            filename_prefix: Prefix for the filename
            
        Returns:
            Path to the saved file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.csv"
            file_path = self.results_folder / filename
            
            df.to_csv(file_path, index=False)
            logger.info(f"DataFrame saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to CSV: {e}")
            return ""


class LoggingSetup:
    """Sets up logging configuration for the application"""
    
    @staticmethod
    def setup_logging(log_level: str = None):
        """
        Set up logging configuration
        
        Args:
            log_level: Logging level (defaults to Config.LOG_LEVEL)
        """
        log_level = log_level or Config.LOG_LEVEL
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pet_food_system.log')
            ]
        )


class InputValidator:
    """Validates user input and provides sanitization"""
    
    @staticmethod
    def validate_user_query(user_input: str) -> tuple[bool, str]:
        """
        Validate user query input
        
        Args:
            user_input: User's query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not user_input or not user_input.strip():
            return False, "Query cannot be empty"
        
        # Check for reasonable length
        if len(user_input.strip()) > 500:
            return False, "Query is too long (max 500 characters)"
        
        # Check for potentially malicious content
        dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        for pattern in dangerous_patterns:
            if pattern.lower() in user_input.lower():
                return False, "Query contains potentially dangerous content"
        
        return True, "Query is valid"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing/replacing invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple consecutive underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:97] + "..."
        
        return filename.strip('_')


class HelpTextProvider:
    """Provides help text and documentation"""
    
    @staticmethod
    def get_help_text() -> str:
        """Get detailed help information"""
        help_text = """
🆘 PET FOOD QUERY SYSTEM - HELP GUIDE
═══════════════════════════════════════════════════════════════════

📋 WHAT YOU CAN ASK:

🐕 Animal-Specific Queries:
  • "dog food"               • "puppy nutrition"
  • "cat food"               • "senior dog food"
  • "kitten food"            • "adult cat dry food"

🥩 Ingredient-Based Queries:
  • "chicken dog food"       • "salmon cat food"
  • "grain-free products"    • "beef ingredients"
  • "no chicken meal"        • "fish-based nutrition"

📊 Nutritional Queries:
  • "high protein food"      • "low fat dog food"
  • "more than 30% protein"  • "less than 15% fat"
  • "high calorie food"      • "fiber-rich nutrition"

🏷️ Brand & Product Queries:
  • "AATU products"          • "Royal Canin dog food"
  • "Hill's Science Diet"    • "specific brand comparison"

🥫 Food Type Queries:
  • "dry food"               • "wet food / canned food"
  • "treats"                 • "prescription diet"
  • "therapeutic food"       • "complete nutrition"

🎯 Combined Queries:
  • "high protein grain-free dog food"
  • "wet kitten food with salmon"
  • "senior cat dry food under 10% fat"
  • "AATU duck products for adult dogs"

💡 TIPS FOR BETTER RESULTS:
  ✅ Be specific about animal type (dog/cat)
  ✅ Include life stage when relevant (puppy/adult/senior)
  ✅ Mention food type preference (dry/wet)
  ✅ Use nutritional terms (high/low protein/fat)
  ✅ Include brand names for specific searches

🔧 SYSTEM COMMANDS:
  • 'help'     - Show this help guide
  • 'quit'     - Exit the system

🔍 EXAMPLE QUERIES:
  You: "high protein dog food"
  System: [Shows products with >25% protein for dogs]
  
  You: "grain-free cat food for seniors"
  System: [Shows grain-free products suitable for senior cats]

═══════════════════════════════════════════════════════════════════
        """
        return help_text


class PerformanceTimer:
    """Simple performance timing utility"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = datetime.now()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = datetime.now()
        if self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time without stopping the timer
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time:
            current_time = datetime.now()
            return (current_time - self.start_time).total_seconds()
        return 0.0


class DataCleaner:
    """Utilities for cleaning and formatting data"""
    
    @staticmethod
    def clean_text_value(value: Any) -> str:
        """
        Clean and format text values from database
        
        Args:
            value: Raw value from database
            
        Returns:
            Cleaned string value
        """
        if value is None or str(value).strip().lower() in ['', 'none', 'null', 'n/a', 'nan']:
            return ""
        
        text = str(value).strip()
        return text
    
    @staticmethod
    def format_percentage(value: str) -> str:
        """
        Format percentage values consistently
        
        Args:
            value: Raw percentage value
            
        Returns:
            Formatted percentage string
        """
        clean_value = DataCleaner.clean_text_value(value)
        if not clean_value:
            return ""
        
        # Ensure percentage sign is present
        if '%' not in clean_value and clean_value.replace('.', '').isdigit():
            clean_value += '%'
        
        return clean_value
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncate text to specified length
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add when truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
