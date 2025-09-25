#!/usr/bin/env python3
"""
Pet Food Query System
A system that takes natural language queries about pet food and generates SQL queries
to retrieve relevant data from the database, then provides intelligent answers using Gemini.
"""

import os
import pandas as pd
import hashlib
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
        
        # Initialize conversation history and caching
        self.conversation_history = []
        self.query_cache = {}
        self.max_history_length = 10
        
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
    
    def get_cache_key(self, user_query: str) -> str:
        """Generate cache key for user query"""
        return hashlib.md5(user_query.lower().encode()).hexdigest()
    
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
        
        # Get conversation context
        context = self._get_conversation_context()
        
        prompt = f"""
        You are a SQL expert specializing in pet food database queries. Generate a PostgreSQL query based on the user's natural language request.
        
        Database Schema:
        {schema_info}
        
        {context}
        
        User Query: "{user_query}"
        
        ENHANCED GUIDELINES:
        1. Use ONLY columns that exist in the schema
        2. Use ILIKE for case-insensitive text searches with % wildcards
        3. Always add LIMIT 15 to prevent large result sets
        4. Order by relevance (id DESC for newest first, or specific criteria)
        5. Return ONLY the SQL query, no explanations or markdown
        6. Handle multiple criteria with AND/OR logic appropriately
        
        ADVANCED PATTERN EXAMPLES:
        
        Animal Type Queries:
        - "dog food" → WHERE target_animal_species ILIKE '%dog%'
        - "cat treats" → WHERE target_animal_species ILIKE '%cat%' AND food_genre ILIKE '%treat%'
        
        Life Stage Queries:
        - "puppy food" → WHERE life_stage ILIKE '%puppy%' OR life_stage ILIKE '%young%'
        - "senior cat" → WHERE target_animal_species ILIKE '%cat%' AND life_stage ILIKE '%senior%'
        
        Nutritional Queries (with safe numeric handling):
        - "high protein" → WHERE (protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%'
        - "low fat" → WHERE (fat_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(fat_percent, '%', '') AS FLOAT) < 10) OR fat_percent ILIKE '%low%'
        - "more than 30% protein" → WHERE protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 30
        
        Ingredient Queries:
        - "chicken" → WHERE ingredients ILIKE '%chicken%' OR type_of_meat ILIKE '%chicken%'
        - "grain free" → WHERE grain_classification ILIKE '%none%' OR grain_classification ILIKE '%grain%free%'
        - "salmon" → WHERE ingredients ILIKE '%salmon%' OR type_of_fish ILIKE '%salmon%'
        
        Brand Queries:
        - "AATU" → WHERE brand_name_english ILIKE '%AATU%' OR brand_name_display ILIKE '%AATU%'
        
        Food Type Queries:
        - "dry food" → WHERE food_type ILIKE '%dry%'
        - "wet food" → WHERE food_type ILIKE '%wet%' OR food_type ILIKE '%can%'
        
        Combined Queries:
        - "high protein dog dry food" → WHERE target_animal_species ILIKE '%dog%' AND food_type ILIKE '%dry%' AND ((protein_percent ~ E'^[0-9]+\\.?[0-9]*%?$' AND CAST(REPLACE(protein_percent, '%', '') AS FLOAT) > 25) OR protein_percent ILIKE '%high%')
        
        CRITICAL NUMERIC HANDLING:
        - Use REPLACE(column, '%', '') to remove % signs before CAST
        - Use regex E'^[0-9]+\\.?[0-9]*%?$' to validate numeric values
        - Always provide fallback with ILIKE for text matches
        - Combine numeric and text searches with OR
        
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
    
    def _get_conversation_context(self) -> str:
        """Get conversation context for better query generation"""
        if not self.conversation_history:
            return ""
        
        recent_queries = self.conversation_history[-3:]  # Last 3 queries
        context = "Recent conversation context:\n"
        for i, item in enumerate(recent_queries, 1):
            context += f"{i}. User asked: \"{item['user_query']}\"\n"
        context += "\nUse this context to better understand the current query.\n"
        return context
    
    def _get_fallback_query(self, user_query: str) -> str:
        """Generate fallback query based on common patterns"""
        query_lower = user_query.lower()
        
        # Basic pattern matching for fallback
        if any(word in query_lower for word in ['dog', 'canine']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE target_animal_species ILIKE '%dog%' LIMIT 15"
        elif any(word in query_lower for word in ['cat', 'feline', 'kitten']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE target_animal_species ILIKE '%cat%' LIMIT 15"
        elif any(word in query_lower for word in ['protein', 'high protein']):
            return f"SELECT * FROM {self.schema['table_name']} WHERE protein_percent ILIKE '%high%' OR ingredients ILIKE '%protein%' LIMIT 15"
        else:
            return f"SELECT * FROM {self.schema['table_name']} ORDER BY id DESC LIMIT 15"
    
    def execute_query(self, sql_query: str, user_query: str = "") -> Optional[pd.DataFrame]:
        """
        Execute SQL query against the database with caching
        
        Args:
            sql_query: SQL query string
            user_query: Original user query for caching
            
        Returns:
            DataFrame with query results or None if error
        """
        # Check cache first
        cache_key = self.get_cache_key(user_query) if user_query else self.get_cache_key(sql_query)
        if cache_key in self.query_cache:
            logger.info("Returning cached results")
            return self.query_cache[cache_key]
        
        try:
            logger.info(f"Executing query: {sql_query}")
            df = pd.read_sql(text(sql_query), self.engine)
            logger.info(f"Query returned {len(df)} rows")
            
            # Cache the results
            if user_query:
                self.query_cache[cache_key] = df
            
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
        conversation_context = self._get_conversation_context()
        
        prompt = f"""
        You are an expert pet nutritionist and veterinarian with deep knowledge of pet food products. Based on the database query results below, provide a comprehensive, helpful, and professionally formatted answer.
        
        User Question: "{user_query}"
        
        {conversation_context}
        
        Database Results ({len(query_results)} products found):
        {results_summary}
        
        ENHANCED INSTRUCTIONS:
        1. 📋 Start with a brief summary of what was found
        2. 🏆 Highlight the TOP 2-3 most relevant products with clear reasoning
        3. 📊 Include specific nutritional data (protein %, fat %, etc.) when available
        4. 🏷️ Mention brand names, product names, and variations clearly
        5. 💡 Provide nutritional insights and recommendations based on the data
        6. ⚖️ Compare products when multiple options exist, highlighting key differences
        7. 🎯 Address the specific user query directly
        8. 🐕/🐱 Consider animal species, life stage, and special needs
        9. ✅ Use bullet points and clear formatting for readability
        10. 💬 Maintain a friendly, professional, and trustworthy tone
        
        FORMAT EXAMPLE:
        "I found [X] products matching your query for [specific need].
        
        🏆 TOP RECOMMENDATIONS:
        • **Brand Name - Product Name**: [Brief description, key nutrition facts]
        • **Brand Name - Product Name**: [Brief description, key nutrition facts]
        
        📊 NUTRITIONAL COMPARISON:
        [Comparison table or key differences]
        
        💡 EXPERT INSIGHTS:
        [Professional advice, considerations, recommendations]"
        
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
        Format DataFrame results in a readable way for the LLM
        
        Args:
            df: Query results DataFrame
            
        Returns:
            Formatted string representation
        """
        if df.empty:
            return "No results found."
        
        # Select most relevant columns for the summary with better organization
        priority_cols = [
            'brand_name_english', 'product_name_english', 'variation_name_final',
            'target_animal_species', 'food_type', 'life_stage'
        ]
        
        nutrition_cols = [
            'protein_percent', 'fat_percent', 'carbohydrates_percent', 
            'metabolizable_energy_100g', 'fiber_label'
        ]
        
        other_cols = [
            'ingredients', 'type_of_meat', 'type_of_fish', 'grain_classification',
            'specific_physical_condition', 'therapeutic_food_category'
        ]
        
        # Only include columns that exist
        available_priority = [col for col in priority_cols if col in df.columns]
        available_nutrition = [col for col in nutrition_cols if col in df.columns]
        available_other = [col for col in other_cols if col in df.columns]
        
        summary_df = df.head(10)  # Limit to first 10 results
        
        result_text = "PRODUCT DATA SUMMARY:\n" + "="*50 + "\n\n"
        
        for idx, row in summary_df.iterrows():
            result_text += f"🔸 PRODUCT #{idx + 1}:\n"
            
            # Basic info
            result_text += "  📋 BASIC INFO:\n"
            for col in available_priority:
                if pd.notna(row[col]) and str(row[col]).strip():
                    clean_name = col.replace('_', ' ').title()
                    result_text += f"    • {clean_name}: {row[col]}\n"
            
            # Nutritional info
            if available_nutrition:
                result_text += "  📊 NUTRITION:\n"
                for col in available_nutrition:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        clean_name = col.replace('_', ' ').replace('percent', '%').title()
                        result_text += f"    • {clean_name}: {row[col]}\n"
            
            # Additional info (truncated for brevity)
            for col in available_other[:2]:  # Only show first 2 to avoid overwhelming
                if pd.notna(row[col]) and str(row[col]).strip():
                    clean_name = col.replace('_', ' ').title()
                    value = str(row[col])
                    if len(value) > 80:
                        value = value[:80] + "..."
                    result_text += f"    • {clean_name}: {value}\n"
            
            result_text += "\n"
        
        if len(df) > 10:
            result_text += f"... and {len(df) - 10} more products in the full dataset.\n"
        
        return result_text
    
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
        
        # Check cache first for exact matches
        cache_key = self.get_cache_key(user_input)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            logger.info("✅ Returning cached result")
            
            # Still generate fresh answer for cached data
            answer = self.generate_final_answer(user_input, cached_result)
            
            result = {
                "user_query": user_input,
                "sql_query": "CACHED_QUERY",
                "results": cached_result,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "cached": True,
                "processing_time": 0.1
            }
            
            self._add_to_conversation_history(result)
            return result
        
        try:
            # Step 1: Generate SQL query
            logger.info("📝 Generating SQL query...")
            sql_query = self.generate_sql_query(user_input)
            
            # Step 2: Execute query
            logger.info("🔄 Executing database query...")
            results = self.execute_query(sql_query, user_input)
            
            if results is None:
                error_result = {
                    "user_query": user_input,
                    "sql_query": sql_query,
                    "results": None,
                    "answer": "Sorry, there was an error executing the database query. Please try rephrasing your question.",
                    "timestamp": datetime.now().isoformat(),
                    "error": True,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
                self._add_to_conversation_history(error_result)
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
            
            # Add to conversation history
            self._add_to_conversation_history(result)
            
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
            self._add_to_conversation_history(error_result)
            return error_result
    
    def _add_to_conversation_history(self, result: Dict[str, Any]):
        """Add query result to conversation history"""
        self.conversation_history.append({
            "user_query": result["user_query"],
            "answer": result["answer"],
            "timestamp": result["timestamp"],
            "result_count": result.get("result_count", 0)
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation"""
        if not self.conversation_history:
            return "No previous queries in this session."
        
        summary = f"📚 Conversation History ({len(self.conversation_history)} queries):\n\n"
        for i, item in enumerate(self.conversation_history[-5:], 1):  # Last 5 queries
            summary += f"{i}. Q: \"{item['user_query']}\"\n"
            summary += f"   Results: {item.get('result_count', 0)} products\n"
            summary += f"   Time: {item['timestamp'][:19]}\n\n"
        
        return summary
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("🗑️ Query cache cleared")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("🗑️ Conversation history cleared")
    
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
  • 'history'  - View conversation history
  • 'clear'    - Clear cache and history
  • 'quit'     - Exit the system

🔍 EXAMPLE CONVERSATION:
  You: "high protein dog food"
  System: [Shows products with >25% protein for dogs]
  
  You: "what about grain-free options?"
  System: [Uses context to find grain-free high-protein dog foods]

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
        print("  • 'history' - Show conversation history")
        print("  • 'clear' - Clear cache and history")
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
            
            if user_input.lower() == 'history':
                print(system.get_conversation_summary())
                continue
            
            if user_input.lower() == 'clear':
                system.clear_cache()
                system.clear_history()
                print("✅ Cache and history cleared!")
                continue
            
            if not user_input:
                print("💭 Please enter a question about pet food, or type 'help' for guidance.")
                continue
            
            # Process the query
            print("⏳ Processing your query...")
            result = system.query(user_input)
            
            # Display results with enhanced formatting
            print(f"\n📊 SQL Query: {result['sql_query']}")
            if result.get('cached'):
                print("⚡ (Retrieved from cache)")
            
            print("\n🎯 Answer:")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)
            
            # Show performance metrics
            if not result.get('error'):
                print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"📊 Results found: {result.get('result_count', 0)} products")
            
            # Optionally show raw data
            print("\n🔍 Options:")
            print("   • (y) Show detailed raw data")
            print("   • (t) Show table comparison view")
            print("   • (h) Show conversation history")
            print("   • (Enter) Continue to next query")
            choice = input("   Your choice: ").strip().lower()
            
            if choice == 'y' and result.get('results') is not None and not result['results'].empty:
                system._display_raw_results(result['results'])
            elif choice == 't' and result.get('results') is not None and not result['results'].empty:
                system._display_table_view(result['results'])
            elif choice == 'h':
                print(system.get_conversation_summary())
    
    except Exception as e:
        print(f"❌ Error initializing system: {e}")

if __name__ == "__main__":
    main()
