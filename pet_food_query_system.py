#!/usr/bin/env python3
"""
Pet Food Query System - Main Application

A modular, clean architecture implementation of the pet food query system
that uses Google Gemini AI to generate SQL queries and provide intelligent
answers about pet food nutrition and products.
"""

from typing import Dict, Any
import logging

# Import our modular components
from config import Config
from query_processor import QueryProcessor
from display import InteractiveInterface
from utils import LoggingSetup, HelpTextProvider, InputValidator

# Set up logging
LoggingSetup.setup_logging()
logger = logging.getLogger(__name__)


class PetFoodQuerySystem:
    """
    Main Pet Food Query System class - now using modular architecture
    
    This class serves as the main interface and coordinates between
    the various specialized modules for clean separation of concerns.
    """
    
    def __init__(self, database_url: str = None, gemini_api_key: str = None):
        """
        Initialize the Pet Food Query System with modular components
        
        Args:
            database_url: PostgreSQL connection string (optional, uses config default)
            gemini_api_key: Google Gemini API key (optional, uses config default)
        """
        # Use provided values or fall back to configuration
        self.database_url = database_url or Config.DATABASE_URL
        self.gemini_api_key = gemini_api_key or Config.GEMINI_API_KEY
        
        # Validate configuration
        is_valid, error_msg = Config.validate_config()
        if not is_valid:
            raise ValueError(f"Configuration error: {error_msg}")
        
        # Set up directories
        Config.setup_directories()
        
        # Initialize core components
        self.query_processor = QueryProcessor(self.database_url, self.gemini_api_key)
        self.interface = InteractiveInterface()
        
        logger.info("Pet Food Query System initialized with modular architecture")
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to process a user query end-to-end
        
        Args:
            user_input: Natural language query from user
            
        Returns:
            Dictionary with query results and answer (for backward compatibility)
        """
        result = self.query_processor.process_query(user_input)
        return result.to_dict()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
            
        Returns:
            Dictionary with system status information
        """
        return self.query_processor.get_system_status()
    
    def run_test_query(self, test_query: str = None) -> Dict[str, Any]:
        """
        Run a test query to verify system functionality
        
        Args:
            test_query: Optional test query
            
        Returns:
            Test result dictionary
        """
        result = self.query_processor.test_query(test_query)
        return result.to_dict()
    def run_interactive_mode(self) -> None:
        """
        Run the system in interactive mode with user interface
        """
        try:
            # Show welcome message
            self.interface.show_welcome_message()
            
            # Main interactive loop
            while True:
                user_input = self.interface.get_user_input()
                
                # Handle system commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.interface.show_goodbye_message()
                    break
                
                if user_input.lower() == 'help':
                    print(HelpTextProvider.get_help_text())
                    continue
                
                if not user_input:
                    self.interface.show_info_message("Please enter a question about pet food, or type 'help' for guidance.")
                    continue
                
                # Validate input
                is_valid, validation_error = InputValidator.validate_user_query(user_input)
                if not is_valid:
                    self.interface.show_error_message(validation_error)
                    continue
                
                # Process the query
                self.interface.show_processing_message()
                result = self.query_processor.process_query(user_input)
                
                # Display results
                self.interface.display_query_result(result.to_dict())
                
                # Handle additional display options
                choice = self.interface.show_options_menu()
                self.interface.handle_data_display_choice(choice, result.to_dict())
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Pet Food Query System! Goodbye!")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            self.interface.show_error_message(f"System error: {e}")


def main():
    """
    Main function to run the Pet Food Query System
    """
    try:
        print("🚀 Initializing Pet Food Query System...")
        
        # Initialize the system with configuration validation
        system = PetFoodQuerySystem()
        print("✅ System initialized successfully!")
        
        # Run in interactive mode
        system.run_interactive_mode()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("   Please check your environment variables:")
        print("   - GEMINI_API_KEY: Your Google Gemini API key")
        print("   - DATABASE_URL: PostgreSQL connection string (optional)")
        return
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        logger.error(f"System initialization failed: {e}")


if __name__ == "__main__":
    main()
