#!/usr/bin/env python3
"""AI service integration for Pet Food Query System using Google Gemini"""

import pandas as pd
from typing import Optional
import logging

try:
    import google.generativeai as genai  # pyright: ignore[reportMissingImports]
except ImportError:
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")
    genai = None

from config import Config
from models import DatabaseSchema, DEFAULT_SCHEMA
from database import SQLQueryBuilder

# Set up logging
logger = logging.getLogger(__name__)


class GeminiAIService:
    """Service for interacting with Google Gemini AI"""
    
    def __init__(self, api_key: str, model_name: str = None, schema: DatabaseSchema = None):
        """
        Initialize the Gemini AI Service
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name to use
            schema: Database schema for query generation
        """
        if genai is None:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        
        self.api_key = api_key
        self.model_name = model_name or Config.GEMINI_MODEL
        self.schema = schema or DEFAULT_SCHEMA
        self.query_builder = SQLQueryBuilder(self.schema)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate_sql_query(self, user_query: str) -> str:
        """
        Generate SQL query from natural language using Gemini
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Generated SQL query string
        """
        schema_info = self.query_builder.get_schema_info()
        query_patterns = self.query_builder.get_query_patterns()
        
        prompt = f"""
        You are a SQL expert specializing in pet food database queries. Generate a PostgreSQL query based on the user's natural language request.
        
        Database Schema:
        {schema_info}
        
        User Query: "{user_query}"
        
        ENHANCED GUIDELINES:
        1. ALWAYS use SELECT * to retrieve ALL columns from the table
        2. Use ONLY column names that exist in the schema for WHERE clauses
        3. Use ILIKE for case-insensitive text searches with % wildcards
        4. Always add LIMIT {Config.MAX_RESULTS_LIMIT} to prevent large result sets
        5. Order by relevance (id DESC for newest first, or specific criteria)
        6. Return ONLY the SQL query, no explanations or markdown
        7. Handle multiple criteria with AND/OR logic appropriately
        8. CRITICAL: Never use SELECT with specific column names - always use SELECT *
        
        {query_patterns}
        
        SQL Query:
        """
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Clean up the response (remove code blocks if present)
            sql_query = self._clean_sql_response(sql_query)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            # Return a fallback query
            return self._get_fallback_query(user_query)
    
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
    
    def _clean_sql_response(self, sql_query: str) -> str:
        """Clean up AI-generated SQL response"""
        # Remove code blocks if present
        if sql_query.startswith('```'):
            lines = sql_query.split('\n')
            sql_query = '\n'.join(lines[1:])
        if sql_query.endswith('```'):
            lines = sql_query.split('\n')
            sql_query = '\n'.join(lines[:-1])
        
        # Remove any trailing semicolon and whitespace
        sql_query = sql_query.strip().rstrip(';')
        
        return sql_query
    
    def _get_fallback_query(self, user_query: str) -> str:
        """Generate fallback query when AI fails"""
        from database import DatabaseManager
        db_manager = DatabaseManager("", self.schema)  # Empty URL since we're only using fallback method
        return db_manager.get_fallback_query(user_query)
    
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
        column_categories = self.schema.get_column_categories()
        
        # Filter to only include columns that actually exist in the retrieved data
        available_basic = [col for col in column_categories["basic_info"] if col in all_columns]
        available_nutrition = [col for col in column_categories["nutritional"] if col in all_columns]
        available_ingredients = [col for col in column_categories["ingredients"] if col in all_columns]
        available_status = [col for col in column_categories["status_and_other"] if col in all_columns]
        
        # Include any remaining columns not categorized above
        categorized_cols = set(available_basic + available_nutrition + available_ingredients + available_status)
        remaining_cols = [col for col in all_columns if col not in categorized_cols]
        
        summary_df = df.head(Config.MAX_DISPLAY_PRODUCTS)  # Show configurable number of products
        
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
        
        if len(df) > Config.MAX_DISPLAY_PRODUCTS:
            result_text += f"📋 NOTE: Showing first {Config.MAX_DISPLAY_PRODUCTS} products out of {len(df)} total results.\n"
            result_text += "ALL COLUMNS retrieved from the database query have been included for comprehensive analysis.\n"
        
        result_text += f"\n🔍 TOTAL COLUMNS RETRIEVED: {len(all_columns)}\n"
        result_text += f"📊 COLUMNS INCLUDED: {', '.join(all_columns)}\n"
        
        return result_text
