#!/usr/bin/env python3
"""Pet Food Query System"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
try:
    import google.generativeai as genai  # pyright: ignore[reportMissingImports]
except ImportError:
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")
    genai = None
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class PetFoodQuerySystem:
    def __init__(self, database_url: str, gemini_api_key: str):
        """
        Initialize the Pet Food Query System
        
        Args:
            database_url: PostgreSQL connection string
            gemini_api_key: Google Gemini API key
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # Configure Gemini
        if genai is None:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")

        gemini_api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Good middle-ground model
        
        # Create results folder if it doesn't exist
        self.results_folder = Path("results")
        self.results_folder.mkdir(exist_ok=True)
        
        # Database schema
        self.schema = {
            "table_name": "pet_food",
            "columns": [
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
        }
        
        # Common query patterns and examples
        self.query_examples = {
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
    
    def validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety and correctness
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Remove whitespace and convert to lowercase for checking
        query_lower = sql_query.lower().strip()
        
        # Security checks - block dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'update', 'insert', 'alter', 'create',
            'truncate', 'grant', 'revoke', 'exec', 'execute'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False, f"Dangerous SQL keyword '{keyword}' detected"
        
        # Must be a SELECT query
        if not query_lower.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        # Check for valid table name
        if self.schema['table_name'] not in query_lower:
            return False, f"Query must reference table '{self.schema['table_name']}'"
        
        # Basic syntax validation
        if query_lower.count('(') != query_lower.count(')'):
            return False, "Mismatched parentheses in query"
        
        return True, "Query is valid"
    
    
    def generate_sql_query(self, user_query: str) -> str:
        """
        Generate SQL query from natural language using Gemini
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Generated SQL query string
        """
        schema_info = f"""
        Table: {self.schema['table_name']}
        Columns: {', '.join(self.schema['columns'])}
        
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
        
        
        prompt = f"""
        You are a SQL expert specializing in pet food database queries. Generate a PostgreSQL query based on the user's natural language request.
        
        Database Schema:
        {schema_info}
        
        User Query: "{user_query}"
        
        ENHANCED GUIDELINES:
        1. ALWAYS use SELECT * to retrieve ALL columns from the table
        2. Use ONLY column names that exist in the schema for WHERE clauses
        3. Use ILIKE for case-insensitive text searches with % wildcards
        4. Always add LIMIT 15 to prevent large result sets
        5. Order by relevance (id DESC for newest first, or specific criteria)
        6. Return ONLY the SQL query, no explanations or markdown
        7. Handle multiple criteria with AND/OR logic appropriately
        8. CRITICAL: Never use SELECT with specific column names - always use SELECT *
        
        ADVANCED PATTERN EXAMPLES (ALWAYS USE SELECT *):
        
        Animal Type Queries:
        - "dog food" → SELECT * FROM pet_food WHERE target_animal_species ILIKE '%dog%' LIMIT 15
        - "cat treats" → SELECT * FROM pet_food WHERE target_animal_species ILIKE '%cat%' AND food_genre ILIKE '%treat%' LIMIT 15
        
        Life Stage Queries:
        - "puppy food" → SELECT * FROM pet_food WHERE life_stage ILIKE '%puppy%' OR life_stage ILIKE '%young%' LIMIT 15
        - "senior cat" → SELECT * FROM pet_food WHERE target_animal_species ILIKE '%cat%' AND life_stage ILIKE '%senior%' LIMIT 15
        
        Nutritional Queries (with safe numeric handling):
        - "high protein" → SELECT * FROM pet_food WHERE (protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%' LIMIT 15
        - "low fat" → SELECT * FROM pet_food WHERE (fat_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(fat_percent, '%', '') AS FLOAT) < 10) OR fat_percent ILIKE '%low%' LIMIT 15
        - "more than 30% protein" → SELECT * FROM pet_food WHERE protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 30 LIMIT 15
        
        Ingredient Queries:
        - "chicken" → SELECT * FROM pet_food WHERE ingredients ILIKE '%chicken%' OR type_of_meat ILIKE '%chicken%' LIMIT 15
        - "grain free" → SELECT * FROM pet_food WHERE grain_classification ILIKE '%none%' OR grain_classification ILIKE '%grain%free%' LIMIT 15
        - "salmon" → SELECT * FROM pet_food WHERE ingredients ILIKE '%salmon%' OR type_of_fish ILIKE '%salmon%' LIMIT 15
        
        Brand Queries:
        - "AATU" → SELECT * FROM pet_food WHERE brand_name_english ILIKE '%AATU%' OR brand_name_display ILIKE '%AATU%' LIMIT 15
        
        Food Type Queries:
        - "dry food" → SELECT * FROM pet_food WHERE food_type ILIKE '%dry%' LIMIT 15
        - "wet food" → SELECT * FROM pet_food WHERE food_type ILIKE '%wet%' OR food_type ILIKE '%can%' LIMIT 15
        
        Combined Queries:
        - "high protein dog dry food" → SELECT * FROM pet_food WHERE target_animal_species ILIKE '%dog%' AND food_type ILIKE '%dry%' AND ((protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%') LIMIT 15
        
        CRITICAL NUMERIC HANDLING:
        - Use REPLACE(column, '%', '') to remove % signs before CAST
        - Use regex E'^[0-9]+\\.?[0-9]*%?$' to validate numeric values
        - Always provide fallback with ILIKE for text matches
        - Combine numeric and text searches with OR
        
        MANDATORY FORMAT:
        Your response must ALWAYS start with "SELECT * FROM pet_food WHERE..." and end with "LIMIT 15"
        NEVER use specific column names in SELECT clause - ALWAYS use SELECT *
        
        SQL Query:
        """
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Clean up the response (remove code blocks if present)
            if sql_query.startswith('```'):
                lines = sql_query.split('\n')
                sql_query = '\n'.join(lines[1:])
            if sql_query.endswith('```'):
                lines = sql_query.split('\n')
                sql_query = '\n'.join(lines[:-1])
            
            # Remove any trailing semicolon and whitespace
            sql_query = sql_query.strip().rstrip(';')
            
            # Validate the generated query
            is_valid, error_msg = self.validate_sql_query(sql_query)
            if not is_valid:
                logger.warning(f"Generated invalid SQL: {error_msg}")
                return self._get_fallback_query(user_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return self._get_fallback_query(user_query)
    
    
    def _get_fallback_query(self, user_query: str) -> str:
        """Generate fallback query based on common patterns - ALWAYS uses SELECT *"""
        query_lower = user_query.lower()
        
        # Basic pattern matching for fallback - all use SELECT * to get ALL columns
        if any(word in query_lower for word in ['dog', 'canine']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE target_animal_species ILIKE '%dog%' LIMIT 15"
        elif any(word in query_lower for word in ['cat', 'feline', 'kitten']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE target_animal_species ILIKE '%cat%' LIMIT 15"
        elif any(word in query_lower for word in ['protein', 'high protein']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE protein_percent ILIKE '%high%' OR ingredients ILIKE '%protein%' LIMIT 15"
        else:
            return f"SELECT * FROM {self.schema['table_name']} ORDER BY id DESC LIMIT 15"
    
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
                    simple_query = f"SELECT * FROM {self.schema['table_name']} LIMIT 15"
                    logger.info("Trying fallback query")
                    df = pd.read_sql(text(simple_query), self.engine)
                    return df
                except Exception:
                    pass
            return None
    
    def generate_final_answer(self, user_query: str, query_results: pd.DataFrame) -> str:
        """
        Generate final answer using Gemini based on query results
        
        Args:
            user_query: Original user query
            query_results: DataFrame with database results
            
        Returns:
            Natural language answer
        """
        if query_results.empty:
            suggestions = self._get_query_suggestions(user_query)
            return f"I couldn't find any relevant pet food data for your query. {suggestions}"
        
        # Convert DataFrame to a more readable format for the LLM
        results_summary = self._format_results_for_llm(query_results)
        
        prompt = f"""
        You are an expert pet nutritionist and veterinarian with deep knowledge of pet food products. You have been provided with COMPLETE DATABASE RESULTS containing ALL columns retrieved from the SQL query. Use this comprehensive dataset to provide the most accurate and thorough analysis possible.
        
        User Question: "{user_query}"
        
        COMPLETE DATABASE RESULTS - ALL RETRIEVED COLUMNS ({len(query_results)} products found):
        {results_summary}
        
        IMPORTANT: The data above contains ALL columns that were retrieved from the database query. This includes:
        • Complete product identification (ID, brand names in multiple languages, product names, variations)
        • Full nutritional analysis (all percentages, energy values, label information)
        • Comprehensive ingredient data (complete lists, meat/fish types, classifications)
        • Status information (review statuses, confirmation status, quality control data)
        • Additional details (therapeutic categories, physical conditions, URLs, content amounts)
        
        COMPREHENSIVE ANALYSIS INSTRUCTIONS:
        1. 📋 Utilize ALL available data fields in your analysis - don't limit to basic info
        2. 🏆 Recommend products based on complete data profiles including status information
        3. 📊 Provide detailed nutritional analysis using ALL nutritional fields available
        4. 🥩 Analyze ingredient composition thoroughly including all classifications
        5. 🎯 Consider ALL targeting factors: species, life stage, activity, health conditions, therapeutic categories
        6. ✅ Reference quality indicators like review status, confirmation status when available
        7. 🔍 Compare products across ALL available dimensions in the retrieved data
        8. 💊 Identify therapeutic applications and special dietary considerations from the data
        9. 📏 Include packaging, content, and availability information when relevant
        10. 💬 Provide expert insights based on the complete product profiles
        
        ENHANCED RESPONSE FORMAT - USE ALL DATA:
        "Based on comprehensive analysis of [X] products with complete database profiles:
        
        🏆 TOP RECOMMENDATIONS (with full data analysis):
        • **[Brand] - [Product] - [Variation]**: 
          - Complete Nutrition: [Use all nutritional fields available]
          - Ingredient Profile: [Use all ingredient and classification data]
          - Target Suitability: [Use life stage, activity level, condition data]
          - Quality Status: [Reference review/confirmation status when available]
          - Additional Info: [Use therapeutic categories, content amounts, etc.]
        
        📊 COMPREHENSIVE NUTRITIONAL ANALYSIS:
        [Detailed comparison using ALL nutritional fields retrieved]
        
        🥩 COMPLETE INGREDIENT BREAKDOWN:
        [Full ingredient analysis using all classification data retrieved]
        
        🎯 TARGETING & SUITABILITY ANALYSIS:
        [Analysis using ALL targeting fields: species, life stage, activity, conditions, therapeutic categories]
        
        ℹ️ QUALITY & STATUS INFORMATION:
        [Reference review status, confirmation status, and other quality indicators when available]
        
        💡 EXPERT VETERINARY INSIGHTS:
        [Professional recommendations based on complete retrieved data profiles]"
        
        CRITICAL: Use ALL the data fields provided in the results above. The system has retrieved complete product information - utilize every relevant piece of data for the most comprehensive analysis possible.
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return self._generate_fallback_answer(user_query, query_results)
    
    def _get_query_suggestions(self, user_query: str) -> str:
        """Generate helpful suggestions when no results are found"""
        suggestions = "\n\n💡 Try these suggestions:\n"
        query_lower = user_query.lower()
        
        if 'dog' in query_lower:
            suggestions += "• 'high protein dog food'\n• 'puppy dry food'\n• 'senior dog nutrition'"
        elif 'cat' in query_lower:
            suggestions += "• 'kitten wet food'\n• 'adult cat dry food'\n• 'senior cat nutrition'"
        else:
            suggestions += "• Specify animal type: 'dog food' or 'cat food'\n"
            suggestions += "• Include life stage: 'puppy', 'adult', 'senior'\n"
            suggestions += "• Mention food type: 'dry food', 'wet food', 'treats'"
        
        return suggestions
    
    def _generate_fallback_answer(self, user_query: str, query_results: pd.DataFrame) -> str:
        """Generate a basic answer when AI response fails"""
        if query_results.empty:
            return "No products found for your query. Please try with different search terms."
        
        answer = f"I found {len(query_results)} products matching your query:\n\n"
        
        # Show top 3 results
        for idx, row in query_results.head(3).iterrows():
            brand = row.get('brand_name_english', 'Unknown Brand')
            product = row.get('product_name_english', 'Unknown Product')
            species = row.get('target_animal_species', '')
            protein = row.get('protein_percent', '')
            
            answer += f"• **{brand} - {product}**"
            if species:
                answer += f" (for {species})"
            if protein:
                answer += f" - Protein: {protein}"
            answer += "\n"
        
        if len(query_results) > 3:
            answer += f"\n...and {len(query_results) - 3} more products."
        
        return answer
    
    def _format_results_for_llm(self, df: pd.DataFrame) -> str:
        """
        Format DataFrame results in a readable way for the LLM using ALL retrieved columns
        
        Args:
            df: Query results DataFrame with ALL columns from SQL query
            
        Returns:
            Formatted string representation with complete data
        """
        if df.empty:
            return "No results found."
        
        # Use ALL columns that were retrieved by the SQL query
        all_columns = df.columns.tolist()
        
        # Organize columns by category for better readability
        basic_info_cols = [
            'id', 'brand_name_display', 'brand_name_english', 'brand_name_kana',
            'empty_col', 'product_name', 'product_name_english', 'product_name_kana',
            'variation_name', 'variation_name_final', 'variation_name_formula',
            'target_animal_species', 'food_genre', 'food_type', 'life_stage'
        ]
        
        nutritional_cols = [
            'protein_percent', 'fat_percent', 'carbohydrates_percent',
            'metabolizable_energy', 'metabolizable_energy_100g',
            'protein_label', 'fat_label', 'fiber_label', 'ash_label', 'moisture_label'
        ]
        
        ingredient_cols = [
            'ingredients', 'type_of_meat', 'type_of_fish', 'legume_classification',
            'grain_classification', 'additives_preservatives'
        ]
        
        status_and_other_cols = [
            'yumika_status', 'final_confirmation_status', 'nutrient_review_status',
            'classification_by_activity_level', 'specific_physical_condition',
            'therapeutic_food_category', 'content_amount_label', 'content_amount_g',
            'product_url'
        ]
        
        # Filter to only include columns that actually exist in the retrieved data
        available_basic = [col for col in basic_info_cols if col in all_columns]
        available_nutrition = [col for col in nutritional_cols if col in all_columns]
        available_ingredients = [col for col in ingredient_cols if col in all_columns]
        available_status = [col for col in status_and_other_cols if col in all_columns]
        
        # Include any remaining columns not categorized above
        categorized_cols = set(available_basic + available_nutrition + available_ingredients + available_status)
        remaining_cols = [col for col in all_columns if col not in categorized_cols]
        
        summary_df = df.head(8)  # Show 8 products for comprehensive data
        
        result_text = "COMPLETE DATABASE RESULTS - ALL RETRIEVED COLUMNS:\n" + "="*70 + "\n\n"
        
        for idx, row in summary_df.iterrows():
            result_text += f"🔸 PRODUCT #{idx + 1}:\n"
            result_text += "─" * 60 + "\n"
            
            # Basic Product Information
            if available_basic:
                result_text += "📋 BASIC PRODUCT INFO:\n"
                for col in available_basic:
                    if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() not in ['', 'none', 'null', 'n/a']:
                        clean_name = col.replace('_', ' ').title()
                        value = str(row[col]).strip()
                        result_text += f"  • {clean_name}: {value}\n"
                result_text += "\n"
            
            # Nutritional Information
            if available_nutrition:
                result_text += "📊 NUTRITIONAL DATA:\n"
                for col in available_nutrition:
                    if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() not in ['', 'none', 'null', 'n/a']:
                        clean_name = col.replace('_', ' ').replace('percent', '%').title()
                        value = str(row[col]).strip()
                        result_text += f"  • {clean_name}: {value}\n"
                result_text += "\n"
            
            # Ingredient Information
            if available_ingredients:
                result_text += "🥩 INGREDIENT COMPOSITION:\n"
                for col in available_ingredients:
                    if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() not in ['', 'none', 'null', 'n/a']:
                        clean_name = col.replace('_', ' ').title()
                        value = str(row[col]).strip()
                        # Truncate very long ingredient lists for readability but keep substantial content
                        if len(value) > 200:
                            value = value[:197] + "..."
                        result_text += f"  • {clean_name}: {value}\n"
                result_text += "\n"
            
            # Status and Additional Information
            if available_status:
                result_text += "ℹ️  STATUS & ADDITIONAL INFO:\n"
                for col in available_status:
                    if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() not in ['', 'none', 'null', 'n/a']:
                        clean_name = col.replace('_', ' ').title()
                        value = str(row[col]).strip()
                        if len(value) > 150:
                            value = value[:147] + "..."
                        result_text += f"  • {clean_name}: {value}\n"
                result_text += "\n"
            
            # Any remaining uncategorized columns
            if remaining_cols:
                result_text += "📝 OTHER RETRIEVED DATA:\n"
                for col in remaining_cols:
                    if pd.notna(row[col]) and str(row[col]).strip() and str(row[col]).strip().lower() not in ['', 'none', 'null', 'n/a']:
                        clean_name = col.replace('_', ' ').title()
                        value = str(row[col]).strip()
                        if len(value) > 100:
                            value = value[:97] + "..."
                        result_text += f"  • {clean_name}: {value}\n"
                result_text += "\n"
            
            result_text += "=" * 60 + "\n\n"
        
        if len(df) > 8:
            result_text += f"📋 NOTE: Showing first 8 products out of {len(df)} total results.\n"
            result_text += "ALL COLUMNS retrieved from the database query have been included for comprehensive analysis.\n"
        
        result_text += f"\n🔍 TOTAL COLUMNS RETRIEVED: {len(all_columns)}\n"
        result_text += f"📊 COLUMNS INCLUDED: {', '.join(all_columns)}\n"
        
        return result_text
    
    def _save_result_to_file(self, result: Dict[str, Any]) -> str:
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
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to process a user query end-to-end
        
        Args:
            user_input: Natural language query from user
            
        Returns:
            Dictionary with query results and answer
        """
        start_time = datetime.now()
        logger.info(f"🔍 Processing query: '{user_input}'")
        
        try:
            # Step 1: Generate SQL query
            logger.info("📝 Generating SQL query...")
            sql_query = self.generate_sql_query(user_input)
            
            # Step 2: Execute query
            logger.info("🔄 Executing database query...")
            results = self.execute_query(sql_query)
            
            if results is None:
                error_result = {
                    "user_query": user_input,
                    "sql_query": sql_query,
                    "results": None,
                    "answer": "Sorry, there was an error executing the database query. Please try rephrasing your question.",
                    "timestamp": datetime.now().isoformat(),
                    "error": True,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "error_message": "Database query execution failed"
                }
                # Save error result to file
                saved_file = self._save_result_to_file(error_result)
                if saved_file:
                    error_result["saved_file"] = saved_file
                return error_result
            
            # Step 3: Generate final answer
            logger.info("🤖 Generating intelligent answer...")
            answer = self.generate_final_answer(user_input, results)
            
            # Prepare final result
            result = {
                "user_query": user_input,
                "sql_query": sql_query,
                "results": results,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Save result to file
            saved_file = self._save_result_to_file(result)
            if saved_file:
                result["saved_file"] = saved_file
            
            logger.info(f"✅ Query completed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in query processing: {e}")
            error_result = {
                "user_query": user_input,
                "sql_query": "ERROR",
                "results": None,
                "answer": "I encountered an unexpected error while processing your query. Please try again with a different question.",
                "timestamp": datetime.now().isoformat(),
                "error": True,
                "error_message": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            # Save error result to file
            saved_file = self._save_result_to_file(error_result)
            if saved_file:
                error_result["saved_file"] = saved_file
            return error_result
    
    
    def _show_help(self) -> str:
        """Show detailed help information"""
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
    
    def _display_raw_results(self, df: pd.DataFrame):
        """Display raw results in a well-formatted, readable way"""
        print(f"\n{'='*80}")
        print(f"📋 RAW DATABASE RESULTS ({len(df)} products found)")
        print(f"{'='*80}")
        
        # Select most important columns for display
        display_cols = [
            'id', 'brand_name_english', 'product_name_english', 'variation_name_final',
            'target_animal_species', 'food_type', 'life_stage', 'protein_percent',
            'fat_percent', 'carbohydrates_percent', 'metabolizable_energy_100g'
        ]
        
        # Only include columns that exist in the dataframe
        available_cols = [col for col in display_cols if col in df.columns]
        display_df = df[available_cols].head(10)  # Show first 10 rows
        
        # Display each product as a card
        for idx, (_, row) in enumerate(display_df.iterrows(), 1):
            print(f"\n🔸 PRODUCT #{idx}")
            print(f"{'─'*60}")
            
            # Group related information
            basic_info = {}
            nutrition_info = {}
            
            for col in available_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    clean_name = col.replace('_', ' ').title()
                    value = str(row[col]).strip()
                    
                    # Categorize the information
                    if col in ['protein_percent', 'fat_percent', 'carbohydrates_percent', 'metabolizable_energy_100g']:
                        nutrition_info[clean_name] = value
                    else:
                        basic_info[clean_name] = value
            
            # Display basic information
            if basic_info:
                print("📋 BASIC INFORMATION:")
                for key, value in basic_info.items():
                    # Truncate long values for better display
                    if len(value) > 60:
                        value = value[:57] + "..."
                    print(f"   • {key:<25}: {value}")
            
            # Display nutritional information
            if nutrition_info:
                print("\n📊 NUTRITIONAL DATA:")
                for key, value in nutrition_info.items():
                    print(f"   • {key:<25}: {value}")
        
        if len(df) > 10:
            print(f"\n{'─'*60}")
            print(f"📋 Note: Showing first 10 products out of {len(df)} total results")
            print("   Use more specific queries to narrow down results")
        
        # Option to save results
        print(f"\n{'─'*60}")
        save_choice = input("💾 Save these results to CSV file? (y/N): ").strip().lower()
        if save_choice == 'y':
            filename = f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Results saved to: {filename}")
        
        print(f"{'='*80}")
    
    def _display_table_view(self, df: pd.DataFrame):
        """Display results in a compact table format for easy comparison"""
        print(f"\n{'='*120}")
        print(f"📊 TABLE COMPARISON VIEW ({len(df)} products)")
        print(f"{'='*120}")
        
        # Select key columns for comparison
        comparison_cols = [
            'brand_name_english', 'product_name_english', 'target_animal_species',
            'food_type', 'protein_percent', 'fat_percent', 'carbohydrates_percent'
        ]
        
        # Only include columns that exist
        available_cols = [col for col in comparison_cols if col in df.columns]
        comparison_df = df[available_cols].head(8)  # Show 8 for better table width
        
        if comparison_df.empty:
            print("No data available for table view.")
            return
        
        # Create a more readable table format
        print(f"\n{'#':<3} {'BRAND':<15} {'PRODUCT':<25} {'ANIMAL':<8} {'TYPE':<8} {'PROTEIN':<8} {'FAT':<6} {'CARBS':<6}")
        print("─" * 120)
        
        for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
            brand = str(row.get('brand_name_english', ''))[:14]
            product = str(row.get('product_name_english', ''))[:24]
            animal = str(row.get('target_animal_species', ''))[:7]
            food_type = str(row.get('food_type', ''))[:7]
            protein = str(row.get('protein_percent', ''))[:7]
            fat = str(row.get('fat_percent', ''))[:5]
            carbs = str(row.get('carbohydrates_percent', ''))[:5]
            
            print(f"{idx:<3} {brand:<15} {product:<25} {animal:<8} {food_type:<8} {protein:<8} {fat:<6} {carbs:<6}")
        
        if len(df) > 8:
            print("─" * 120)
            print(f"📋 Showing first 8 products out of {len(df)} total results")
        
        # Enhanced nutritional summary
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
                        if clean_val and clean_val.replace('.', '').isdigit():
                            numeric_values.append(float(clean_val))
                    except Exception:
                        continue
                
                if numeric_values:
                    col_name = col.replace('_percent', '').replace('_', ' ').title()
                    avg_val = sum(numeric_values) / len(numeric_values)
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    print(f"   • {col_name:<12}: Average {avg_val:.1f}%  |  Range {min_val:.1f}% - {max_val:.1f}%")
        
        print(f"{'='*120}")

def main():
    """
    Main function to run the Pet Food Query System
    """
    # Configuration
    DATABASE_URL = 'postgresql://pguser:pguser@localhost:5454/nutrient_food'
    
    # Get Gemini API key from environment variable
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        print("❌ Error: Please set GEMINI_API_KEY environment variable")
        print("   Example: export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Initialize the system
    try:
        print("🚀 Initializing Pet Food Query System...")
        system = PetFoodQuerySystem(DATABASE_URL, GEMINI_API_KEY)
        print("✅ System initialized successfully!")
        
        # Interactive loop
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
        
        while True:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Pet Food Query System! Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print(system._show_help())
                continue
            
            if not user_input:
                print("💭 Please enter a question about pet food, or type 'help' for guidance.")
                continue
            
            # Process the query
            print("⏳ Processing your query...")
            result = system.query(user_input)
            
            # Display results with enhanced formatting
            print(f"\n📊 SQL Query: {result['sql_query']}")
            
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
            
            # Optionally show raw data
            print("\n🔍 Options:")
            print("   • (y) Show detailed raw data")
            print("   • (t) Show table comparison view")
            print("   • (Enter) Continue to next query")
            choice = input("   Your choice: ").strip().lower()
            
            if choice == 'y' and result.get('results') is not None and not result['results'].empty:
                system._display_raw_results(result['results'])
            elif choice == 't' and result.get('results') is not None and not result['results'].empty:
                system._display_table_view(result['results'])
    
    except Exception as e:
        print(f"❌ Error initializing system: {e}")

if __name__ == "__main__":
    main()
