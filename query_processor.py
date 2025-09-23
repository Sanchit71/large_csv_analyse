#!/usr/bin/env python3
"""Query processing logic for Pet Food Query System"""

from datetime import datetime
from typing import Dict, Any
import logging

from config import Config
from models import QueryResult, DatabaseSchema, DEFAULT_SCHEMA
from database import DatabaseManager
from ai_service import GeminiAIService
from utils import FileManager, PerformanceTimer, InputValidator

# Set up logging
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Main query processing engine that coordinates all components"""
    
    def __init__(self, 
                 database_url: str, 
                 gemini_api_key: str,
                 schema: DatabaseSchema = None,
                 file_manager: FileManager = None):
        """
        Initialize the Query Processor
        
        Args:
            database_url: PostgreSQL connection string
            gemini_api_key: Google Gemini API key
            schema: Database schema definition
            file_manager: File manager for saving results
        """
        self.schema = schema or DEFAULT_SCHEMA
        self.file_manager = file_manager or FileManager()
        
        # Initialize components
        self.db_manager = DatabaseManager(database_url, self.schema)
        self.ai_service = GeminiAIService(gemini_api_key, Config.GEMINI_MODEL, self.schema)
        
        # Test connections
        self._validate_connections()
    
    def _validate_connections(self) -> None:
        """Validate database and AI service connections"""
        # Test database connection
        db_connected, db_error = self.db_manager.test_connection()
        if not db_connected:
            logger.warning(f"Database connection issue: {db_error}")
        
        # AI service validation is done during initialization
        logger.info("Query processor initialized successfully")
    
    def process_query(self, user_input: str) -> QueryResult:
        """
        Main method to process a user query end-to-end
        
        Args:
            user_input: Natural language query from user
            
        Returns:
            QueryResult object with complete processing results
        """
        timer = PerformanceTimer()
        timer.start()
        
        logger.info(f"🔍 Processing query: '{user_input}'")
        
        # Validate input
        is_valid, validation_error = InputValidator.validate_user_query(user_input)
        if not is_valid:
            return QueryResult.create_error_result(
                user_query=user_input,
                sql_query="VALIDATION_ERROR",
                error_message=validation_error,
                processing_time=timer.stop()
            )
        
        try:
            # Step 1: Generate SQL query
            logger.info("📝 Generating SQL query...")
            sql_query = self.ai_service.generate_sql_query(user_input)
            
            # Validate the generated SQL
            validation_result = self.db_manager.validate_sql_query(sql_query)
            if not validation_result.is_valid:
                logger.warning(f"Generated invalid SQL: {validation_result.error_message}")
                sql_query = self.db_manager.get_fallback_query(user_input)
            
            # Step 2: Execute query
            logger.info("🔄 Executing database query...")
            results_df = self.db_manager.execute_query(sql_query)
            
            if results_df is None:
                error_result = QueryResult.create_error_result(
                    user_query=user_input,
                    sql_query=sql_query,
                    error_message="Database query execution failed",
                    processing_time=timer.stop()
                )
                # Save error result to file
                saved_file = self.file_manager.save_result_to_file(error_result.to_dict())
                if saved_file:
                    error_result.saved_file = saved_file
                return error_result
            
            # Step 3: Generate final answer
            logger.info("🤖 Generating intelligent answer...")
            answer = self.ai_service.generate_final_answer(user_input, results_df)
            
            # Create result object
            result = QueryResult(
                user_query=user_input,
                sql_query=sql_query,
                results=results_df,
                answer=answer,
                timestamp=datetime.now().isoformat(),
                result_count=len(results_df),
                processing_time=timer.stop()
            )
            
            # Save result to file
            saved_file = self.file_manager.save_result_to_file(result.to_dict())
            if saved_file:
                result.saved_file = saved_file
            
            logger.info(f"✅ Query completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in query processing: {e}")
            error_result = QueryResult.create_error_result(
                user_query=user_input,
                sql_query=sql_query if 'sql_query' in locals() else "ERROR",
                error_message=str(e),
                processing_time=timer.stop()
            )
            # Save error result to file
            saved_file = self.file_manager.save_result_to_file(error_result.to_dict())
            if saved_file:
                error_result.saved_file = saved_file
            return error_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information
        
        Returns:
            Dictionary with system status
        """
        db_connected, db_message = self.db_manager.test_connection()
        
        return {
            "database_connected": db_connected,
            "database_message": db_message,
            "ai_service_configured": self.ai_service.model is not None,
            "schema_table": self.schema.table_name,
            "schema_columns_count": len(self.schema.columns),
            "results_folder": str(self.file_manager.results_folder),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_query(self, test_query: str = None) -> QueryResult:
        """
        Run a test query to verify system functionality
        
        Args:
            test_query: Optional test query (defaults to a simple query)
            
        Returns:
            QueryResult from the test
        """
        if test_query is None:
            test_query = "show me some dog food"
        
        logger.info(f"Running test query: {test_query}")
        return self.process_query(test_query)


class BatchQueryProcessor:
    """Processes multiple queries in batch"""
    
    def __init__(self, query_processor: QueryProcessor):
        """
        Initialize Batch Query Processor
        
        Args:
            query_processor: Main query processor instance
        """
        self.query_processor = query_processor
    
    def process_batch(self, queries: list[str]) -> list[QueryResult]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            List of QueryResult objects
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing batch query {i}/{len(queries)}: {query}")
            result = self.query_processor.process_query(query)
            results.append(result)
            
            # Small delay between queries to be respectful to the AI service
            if i < len(queries):
                import time
                time.sleep(1)
        
        return results
    
    def save_batch_results(self, results: list[QueryResult], filename_prefix: str = "batch_results") -> str:
        """
        Save batch results to a summary file
        
        Args:
            results: List of QueryResult objects
            filename_prefix: Prefix for the output file
            
        Returns:
            Path to the saved summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.txt"
        file_path = self.query_processor.file_manager.results_folder / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("PET FOOD QUERY SYSTEM - BATCH RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Batch processed: {datetime.now().isoformat()}\n")
                f.write(f"Total queries: {len(results)}\n")
                f.write(f"Successful queries: {sum(1 for r in results if not r.error)}\n")
                f.write(f"Failed queries: {sum(1 for r in results if r.error)}\n\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"\n{'='*60}\n")
                    f.write(f"QUERY #{i}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"User Query: {result.user_query}\n")
                    f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                    f.write(f"Results Found: {result.result_count}\n")
                    f.write(f"Status: {'ERROR' if result.error else 'SUCCESS'}\n")
                    
                    if result.error:
                        f.write(f"Error: {result.error_message}\n")
                    else:
                        f.write(f"\nSQL Query:\n{result.sql_query}\n")
                        f.write(f"\nAnswer:\n{result.answer}\n")
                
                f.write(f"\n{'='*80}\n")
                f.write("End of Batch Results\n")
                f.write("="*80 + "\n")
            
            logger.info(f"Batch results saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
            return ""


class QueryAnalyzer:
    """Analyzes query patterns and provides insights"""
    
    def __init__(self, query_processor: QueryProcessor):
        """
        Initialize Query Analyzer
        
        Args:
            query_processor: Main query processor instance
        """
        self.query_processor = query_processor
    
    def analyze_query_complexity(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a user query
        
        Args:
            user_query: User's natural language query
            
        Returns:
            Dictionary with complexity analysis
        """
        query_lower = user_query.lower()
        
        # Count different types of criteria
        animal_types = sum(1 for word in ['dog', 'cat', 'puppy', 'kitten', 'canine', 'feline'] if word in query_lower)
        nutritional_terms = sum(1 for word in ['protein', 'fat', 'carb', 'calorie', 'fiber', 'vitamin'] if word in query_lower)
        ingredient_terms = sum(1 for word in ['chicken', 'beef', 'salmon', 'grain', 'rice', 'vegetable'] if word in query_lower)
        brand_indicators = sum(1 for word in ['brand', 'aatu', 'royal', 'hill', 'science'] if word in query_lower)
        
        complexity_score = len(query_lower.split()) + animal_types + nutritional_terms + ingredient_terms + brand_indicators
        
        if complexity_score <= 5:
            complexity_level = "Simple"
        elif complexity_score <= 10:
            complexity_level = "Medium"
        else:
            complexity_level = "Complex"
        
        return {
            "query": user_query,
            "word_count": len(query_lower.split()),
            "animal_types_mentioned": animal_types,
            "nutritional_terms": nutritional_terms,
            "ingredient_terms": ingredient_terms,
            "brand_indicators": brand_indicators,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "estimated_processing_time": complexity_score * 0.5  # seconds
        }
    
    def suggest_query_improvements(self, user_query: str) -> list[str]:
        """
        Suggest improvements to make a query more effective
        
        Args:
            user_query: User's natural language query
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        query_lower = user_query.lower()
        
        # Check for missing animal type
        if not any(word in query_lower for word in ['dog', 'cat', 'puppy', 'kitten', 'canine', 'feline']):
            suggestions.append("Consider specifying the animal type (dog, cat, puppy, kitten)")
        
        # Check for vague terms
        if any(word in query_lower for word in ['good', 'best', 'healthy', 'quality']):
            suggestions.append("Replace vague terms like 'good' or 'healthy' with specific nutritional criteria")
        
        # Check for missing specificity
        if len(query_lower.split()) < 3:
            suggestions.append("Add more specific criteria like life stage, food type, or nutritional requirements")
        
        # Check for missing life stage
        if not any(word in query_lower for word in ['puppy', 'kitten', 'adult', 'senior', 'young', 'old']):
            suggestions.append("Consider adding life stage information (puppy, adult, senior)")
        
        # Check for missing food type
        if not any(word in query_lower for word in ['dry', 'wet', 'treat', 'can', 'kibble']):
            suggestions.append("Specify food type preference (dry food, wet food, treats)")
        
        return suggestions
