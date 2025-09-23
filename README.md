# Pet Food Query System

A natural language query system for pet food database that uses LLM to generate SQL queries and provide intelligent answers using Google Gemini.

## Features

- 🔍 Natural language to SQL query generation
- 🗄️ PostgreSQL database integration  
- 🤖 Intelligent answers using Google Gemini (gemini-1.5-flash)
- 🐕 Pet nutrition expertise
- 📊 Interactive command-line interface

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API key:**
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```
   
   Or create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

3. **Ensure your PostgreSQL database is running:**
   - Database: `nutrient_food`
   - Table: `pet_food`
   - Connection: `postgresql://pguser:pguser@localhost:5454/nutrient_food`

## Usage

### Interactive Mode
```bash
python pet_food_query_system.py
```

Then ask questions like:
- "what should i give dog food healthy"
- "best puppy food with high protein"
- "cat food for senior cats"
- "grain-free dog food recommendations"

### Programmatic Usage
```python
from pet_food_query_system import PetFoodQuerySystem

# Initialize
system = PetFoodQuerySystem(
    database_url='postgresql://pguser:pguser@localhost:5454/nutrient_food',
    gemini_api_key='your_api_key'
)

# Query
result = system.query("what should i give dog food healthy")
print(result['answer'])
```

### Example Queries
```bash
python example_usage.py
```

## How It Works

1. **User Input**: Natural language query about pet food
2. **SQL Generation**: Gemini LLM converts the query to SQL using database schema
3. **Database Query**: Execute SQL against PostgreSQL database
4. **Answer Generation**: Gemini provides intelligent answer based on results

## Database Schema

The system works with a `pet_food` table containing 40 columns:

- **Product information**: brand_name_english, product_name_english, variations
- **Target animals**: target_animal_species, life_stage (puppy, adult, senior)
- **Nutritional content**: protein_percent, fat_percent, carbohydrates_percent
- **Content details**: ingredients, content_amount_g, metabolizable_energy
- **Classifications**: food_genre, food_type, therapeutic_food_category

**Important**: All columns except `id` are stored as TEXT. The system automatically:
- Detects numeric values in text columns (e.g., "30.0", "12.5")
- Uses safe type casting with regex validation
- Handles mixed formats (e.g., "350kcal/100g" in energy fields)
- Combines numeric and text-based searches for flexibility

## Configuration

- **Model**: Uses `gemini-1.5-flash` (good balance of quality and speed)
- **Database**: PostgreSQL with SQLAlchemy
- **Results**: Limited to 10-20 most relevant entries per query
