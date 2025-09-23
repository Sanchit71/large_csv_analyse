#!/usr/bin/env python3
"""
Example Usage of the Modular Pet Food Query System

This script demonstrates how to use the refactored pet food query system
with its clean modular architecture.
"""

from pet_food_query_system import PetFoodQuerySystem
from query_processor import BatchQueryProcessor, QueryAnalyzer
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example of basic usage with the modular system"""
    print("🔥 Example: Basic Usage")
    print("=" * 50)
    
    try:
        # Initialize the system (uses configuration from config.py)
        system = PetFoodQuerySystem()
        
        # Single query
        result = system.query("high protein dog food")
        
        print(f"Query: {result['user_query']}")
        print(f"SQL: {result['sql_query']}")
        print(f"Found: {result['result_count']} products")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Answer: {result['answer'][:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example of batch query processing"""
    print("\n🔥 Example: Batch Processing")
    print("=" * 50)
    
    try:
        # Initialize the system
        system = PetFoodQuerySystem()
        
        # Create batch processor
        batch_processor = BatchQueryProcessor(system.query_processor)
        
        # Define multiple queries
        queries = [
            "high protein dog food",
            "grain-free cat food",
            "senior dog nutrition",
            "AATU brand products"
        ]
        
        print(f"Processing {len(queries)} queries in batch...")
        
        # Process all queries
        results = batch_processor.process_batch(queries)
        
        # Save batch results
        summary_file = batch_processor.save_batch_results(results)
        
        print(f"Batch processing completed!")
        print(f"Summary saved to: {summary_file}")
        
        # Show brief summary
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)
        print(f"Successful queries: {successful}")
        print(f"Failed queries: {failed}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_query_analysis():
    """Example of query analysis features"""
    print("\n🔥 Example: Query Analysis")
    print("=" * 50)
    
    try:
        # Initialize the system
        system = PetFoodQuerySystem()
        
        # Create query analyzer
        analyzer = QueryAnalyzer(system.query_processor)
        
        # Analyze different query complexities
        test_queries = [
            "dog food",
            "high protein dog food for seniors",
            "grain-free chicken-based dry food for active adult dogs with sensitive stomach"
        ]
        
        for query in test_queries:
            analysis = analyzer.analyze_query_complexity(query)
            suggestions = analyzer.suggest_query_improvements(query)
            
            print(f"\nQuery: '{query}'")
            print(f"Complexity: {analysis['complexity_level']} (score: {analysis['complexity_score']})")
            print(f"Word count: {analysis['word_count']}")
            print(f"Estimated processing time: {analysis['estimated_processing_time']:.1f}s")
            
            if suggestions:
                print("Suggestions:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
            else:
                print("No suggestions - query looks good!")
    
    except Exception as e:
        print(f"Error: {e}")


def example_system_status():
    """Example of system status monitoring"""
    print("\n🔥 Example: System Status")
    print("=" * 50)
    
    try:
        # Initialize the system
        system = PetFoodQuerySystem()
        
        # Get system status
        status = system.get_system_status()
        
        print("System Status:")
        print(f"  Database connected: {status['database_connected']}")
        print(f"  Database message: {status['database_message']}")
        print(f"  AI service configured: {status['ai_service_configured']}")
        print(f"  Schema table: {status['schema_table']}")
        print(f"  Schema columns: {status['schema_columns_count']}")
        print(f"  Results folder: {status['results_folder']}")
        print(f"  Status timestamp: {status['timestamp']}")
        
        # Run test query
        print("\nRunning test query...")
        test_result = system.run_test_query("test dog food query")
        print(f"Test query result: {'SUCCESS' if not test_result['error'] else 'FAILED'}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_configuration():
    """Example of configuration access"""
    print("\n🔥 Example: Configuration")
    print("=" * 50)
    
    try:
        print("Configuration Settings:")
        print(f"  Database URL: {Config.DATABASE_URL}")
        print(f"  Gemini Model: {Config.GEMINI_MODEL}")
        print(f"  Results Folder: {Config.RESULTS_FOLDER}")
        print(f"  Max Results Limit: {Config.MAX_RESULTS_LIMIT}")
        print(f"  Max Display Products: {Config.MAX_DISPLAY_PRODUCTS}")
        print(f"  Query Timeout: {Config.QUERY_TIMEOUT}s")
        
        # Validate configuration
        is_valid, message = Config.validate_config()
        print(f"  Configuration Valid: {is_valid}")
        if not is_valid:
            print(f"  Error: {message}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples"""
    print("🐕🐱 Pet Food Query System - Modular Architecture Examples")
    print("=" * 70)
    
    # Run all examples
    example_basic_usage()
    example_batch_processing()
    example_query_analysis()
    example_system_status()
    example_configuration()
    
    print("\n✅ All examples completed!")
    print("\n💡 Key Benefits of the Modular Architecture:")
    print("  • Clean separation of concerns")
    print("  • Easy to test individual components")
    print("  • Configurable and extensible")
    print("  • Better error handling and logging")
    print("  • Reusable components")
    print("  • Maintainable codebase")


if __name__ == "__main__":
    main()
