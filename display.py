#!/usr/bin/env python3
"""Display and formatting utilities for Pet Food Query System"""

import pandas as pd
from typing import List, Optional
from datetime import datetime
import logging

from config import Config
from models import DisplayColumns, DEFAULT_DISPLAY_COLUMNS
from utils import DataCleaner, FileManager

# Set up logging
logger = logging.getLogger(__name__)


class DisplayManager:
    """Manages different display modes and formatting for query results"""
    
    def __init__(self, display_columns: DisplayColumns = None, file_manager: FileManager = None):
        """
        Initialize Display Manager
        
        Args:
            display_columns: Column configuration for different display modes
            file_manager: File manager for saving results
        """
        self.display_columns = display_columns or DEFAULT_DISPLAY_COLUMNS
        self.file_manager = file_manager or FileManager()
    
    def display_raw_results(self, df: pd.DataFrame) -> None:
        """Display raw results in a well-formatted, readable way"""
        print(f"\n{'='*80}")
        print(f"📋 RAW DATABASE RESULTS ({len(df)} products found)")
        print(f"{'='*80}")
        
        if df.empty:
            print("No results to display.")
            return
        
        # Select most important columns for display
        available_cols = [col for col in self.display_columns.raw_display if col in df.columns]
        display_df = df[available_cols].head(Config.MAX_RAW_DISPLAY_PRODUCTS)
        
        # Display each product as a card
        for idx, (_, row) in enumerate(display_df.iterrows(), 1):
            print(f"\n🔸 PRODUCT #{idx}")
            print(f"{'─'*60}")
            
            # Group related information
            basic_info = {}
            nutrition_info = {}
            
            for col in available_cols:
                value = DataCleaner.clean_text_value(row[col])
                if value:
                    clean_name = col.replace('_', ' ').title()
                    
                    # Categorize the information
                    if col in ['protein_percent', 'fat_percent', 'carbohydrates_percent', 'metabolizable_energy_100g']:
                        nutrition_info[clean_name] = DataCleaner.format_percentage(value) if 'percent' in col else value
                    else:
                        basic_info[clean_name] = DataCleaner.truncate_text(value, 60)
            
            # Display basic information
            if basic_info:
                print("📋 BASIC INFORMATION:")
                for key, value in basic_info.items():
                    print(f"   • {key:<25}: {value}")
            
            # Display nutritional information
            if nutrition_info:
                print("\n📊 NUTRITIONAL DATA:")
                for key, value in nutrition_info.items():
                    print(f"   • {key:<25}: {value}")
        
        if len(df) > Config.MAX_RAW_DISPLAY_PRODUCTS:
            print(f"\n{'─'*60}")
            print(f"📋 Note: Showing first {Config.MAX_RAW_DISPLAY_PRODUCTS} products out of {len(df)} total results")
            print("   Use more specific queries to narrow down results")
        
        # Option to save results
        print(f"\n{'─'*60}")
        save_choice = input("💾 Save these results to CSV file? (y/N): ").strip().lower()
        if save_choice == 'y':
            saved_file = self.file_manager.save_dataframe_to_csv(df)
            if saved_file:
                print(f"✅ Results saved to: {saved_file}")
        
        print(f"{'='*80}")
    
    def display_table_view(self, df: pd.DataFrame) -> None:
        """Display results in a compact table format for easy comparison"""
        print(f"\n{'='*120}")
        print(f"📊 TABLE COMPARISON VIEW ({len(df)} products)")
        print(f"{'='*120}")
        
        if df.empty:
            print("No data available for table view.")
            return
        
        # Select key columns for comparison
        available_cols = [col for col in self.display_columns.comparison_view if col in df.columns]
        comparison_df = df[available_cols].head(Config.MAX_DISPLAY_PRODUCTS)
        
        # Create a more readable table format
        print(f"\n{'#':<3} {'BRAND':<15} {'PRODUCT':<25} {'ANIMAL':<8} {'TYPE':<8} {'PROTEIN':<8} {'FAT':<6} {'CARBS':<6}")
        print("─" * 120)
        
        for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
            brand = DataCleaner.truncate_text(str(row.get('brand_name_english', '')), 14, '')
            product = DataCleaner.truncate_text(str(row.get('product_name_english', '')), 24, '')
            animal = DataCleaner.truncate_text(str(row.get('target_animal_species', '')), 7, '')
            food_type = DataCleaner.truncate_text(str(row.get('food_type', '')), 7, '')
            protein = DataCleaner.truncate_text(str(row.get('protein_percent', '')), 7, '')
            fat = DataCleaner.truncate_text(str(row.get('fat_percent', '')), 5, '')
            carbs = DataCleaner.truncate_text(str(row.get('carbohydrates_percent', '')), 5, '')
            
            print(f"{idx:<3} {brand:<15} {product:<25} {animal:<8} {food_type:<8} {protein:<8} {fat:<6} {carbs:<6}")
        
        if len(df) > Config.MAX_DISPLAY_PRODUCTS:
            print("─" * 120)
            print(f"📋 Showing first {Config.MAX_DISPLAY_PRODUCTS} products out of {len(df)} total results")
        
        # Enhanced nutritional summary
        self._display_nutritional_summary(df)
        
        print(f"{'='*120}")
    
    def _display_nutritional_summary(self, df: pd.DataFrame) -> None:
        """Display nutritional summary statistics"""
        print(f"\n{'─'*60}")
        print("📊 NUTRITIONAL SUMMARY:")
        
        numeric_cols = ['protein_percent', 'fat_percent', 'carbohydrates_percent']
        for col in numeric_cols:
            if col in df.columns:
                # Try to extract numeric values
                numeric_values = []
                for val in df[col].dropna():
                    try:
                        # Remove % sign and convert to float
                        clean_val = str(val).replace('%', '').strip()
                        if clean_val and clean_val.replace('.', '').replace('-', '').isdigit():
                            numeric_values.append(float(clean_val))
                    except Exception:
                        continue
                
                if numeric_values:
                    col_name = col.replace('_percent', '').replace('_', ' ').title()
                    avg_val = sum(numeric_values) / len(numeric_values)
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    print(f"   • {col_name:<12}: Average {avg_val:.1f}%  |  Range {min_val:.1f}% - {max_val:.1f}%")


class InteractiveInterface:
    """Handles interactive user interface and input/output"""
    
    def __init__(self, display_manager: DisplayManager = None):
        """
        Initialize Interactive Interface
        
        Args:
            display_manager: Display manager for formatting results
        """
        self.display_manager = display_manager or DisplayManager()
    
    def show_welcome_message(self) -> None:
        """Display welcome message and instructions"""
        print("\n" + "=" * 70)
        print("🐕🐱 Enhanced Pet Food Query System Ready!")
        print("Ask me anything about pet food nutrition and products")
        print("=" * 70)
        print("\n💡 Example queries:")
        print("  • 'high protein dog food'")
        print("  • 'grain-free cat food for seniors'")
        print("  • 'AATU brand products'")
        print("  • 'wet food with salmon for kittens'")
        print("\n🔧 Commands:")
        print("  • 'help' - Show detailed help")
        print("  • 'quit' - Exit")
        print("=" * 70)
    
    def get_user_input(self, prompt: str = "\n❓ Your question: ") -> str:
        """
        Get user input with the specified prompt
        
        Args:
            prompt: Input prompt to display
            
        Returns:
            User input string
        """
        return input(prompt).strip()
    
    def show_processing_message(self) -> None:
        """Show processing message"""
        print("⏳ Processing your query...")
    
    def display_query_result(self, result: dict) -> None:
        """
        Display the main query result
        
        Args:
            result: Query result dictionary
        """
        # Display SQL query
        print(f"\n📊 SQL Query: {result['sql_query']}")
        
        # Display AI answer
        print("\n🎯 Answer:")
        print("-" * 50)
        print(result['answer'])
        print("-" * 50)
        
        # Show performance metrics
        if not result.get('error'):
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"📊 Results found: {result.get('result_count', 0)} products")
        
        # Show saved file information
        if result.get('saved_file'):
            print(f"💾 Result saved to: {result['saved_file']}")
    
    def show_options_menu(self) -> str:
        """
        Show options menu and get user choice
        
        Returns:
            User's choice
        """
        print("\n🔍 Options:")
        print("   • (y) Show detailed raw data")
        print("   • (t) Show table comparison view")
        print("   • (Enter) Continue to next query")
        return input("   Your choice: ").strip().lower()
    
    def handle_data_display_choice(self, choice: str, result: dict) -> None:
        """
        Handle user's choice for data display
        
        Args:
            choice: User's choice ('y', 't', or other)
            result: Query result dictionary
        """
        if choice == 'y' and result.get('results') is not None and not result['results'].empty:
            self.display_manager.display_raw_results(result['results'])
        elif choice == 't' and result.get('results') is not None and not result['results'].empty:
            self.display_manager.display_table_view(result['results'])
    
    def show_error_message(self, message: str) -> None:
        """
        Display error message
        
        Args:
            message: Error message to display
        """
        print(f"❌ Error: {message}")
    
    def show_info_message(self, message: str) -> None:
        """
        Display info message
        
        Args:
            message: Info message to display
        """
        print(f"💭 {message}")
    
    def show_goodbye_message(self) -> None:
        """Display goodbye message"""
        print("👋 Thank you for using Pet Food Query System! Goodbye!")


class ResultFormatter:
    """Formats query results for different output modes"""
    
    @staticmethod
    def format_for_console(result: dict) -> str:
        """
        Format result for console display
        
        Args:
            result: Query result dictionary
            
        Returns:
            Formatted string for console
        """
        formatted = f"""
Query: {result['user_query']}
SQL: {result['sql_query']}
Results: {result.get('result_count', 0)} products found
Processing Time: {result.get('processing_time', 0):.2f}s

Answer:
{result['answer']}
"""
        return formatted.strip()
    
    @staticmethod
    def format_for_file(result: dict) -> str:
        """
        Format result for file output
        
        Args:
            result: Query result dictionary
            
        Returns:
            Formatted string for file
        """
        timestamp = result.get('timestamp', datetime.now().isoformat())
        
        formatted = f"""
PET FOOD QUERY SYSTEM - RESULT
{'='*80}

Timestamp: {timestamp}
User Query: {result['user_query']}
Processing Time: {result.get('processing_time', 0):.2f}s
Results Found: {result.get('result_count', 0)} products

SQL Query:
{'-'*40}
{result['sql_query']}

AI Generated Answer:
{'-'*40}
{result['answer']}
"""
        
        if result.get('error'):
            formatted += f"""

Error Information:
{'-'*40}
Error: {result.get('error_message', 'Unknown error')}
"""
        
        formatted += f"\n{'='*80}\nEnd of Result\n{'='*80}"
        
        return formatted.strip()
