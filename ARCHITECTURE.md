# Pet Food Query System - Modular Architecture

## 🏗️ Architecture Overview

The Pet Food Query System has been refactored from a monolithic design into a clean, modular architecture that follows software engineering best practices. This documentation explains the new structure and how to work with it.

## 📁 Module Structure

```
pet_food_query_system/
├── config.py              # Configuration management
├── models.py               # Data models and schemas
├── database.py             # Database operations and SQL handling
├── ai_service.py           # Google Gemini AI integration
├── query_processor.py      # Main query processing logic
├── display.py              # UI and output formatting
├── utils.py                # Utility functions and helpers
├── pet_food_query_system.py # Main application interface
├── example_usage.py        # Usage examples
└── ARCHITECTURE.md         # This documentation
```

## 🧩 Module Responsibilities

### 1. `config.py` - Configuration Management
- **Purpose**: Centralized configuration and environment management
- **Key Classes**: `Config`
- **Responsibilities**:
  - Environment variable management
  - Default configuration values
  - Configuration validation
  - Directory setup

```python
from config import Config

# Access configuration
database_url = Config.DATABASE_URL
api_key = Config.GEMINI_API_KEY

# Validate configuration
is_valid, error = Config.validate_config()
```

### 2. `models.py` - Data Models and Schemas
- **Purpose**: Define data structures and models
- **Key Classes**: `DatabaseSchema`, `QueryResult`, `SQLValidationResult`, `DisplayColumns`
- **Responsibilities**:
  - Database schema definitions
  - Result data structures
  - Validation result models
  - Display configuration models

```python
from models import DatabaseSchema, QueryResult

# Use schema
schema = DatabaseSchema()
columns = schema.get_column_categories()

# Create query result
result = QueryResult(user_query="test", sql_query="SELECT...", ...)
```

### 3. `database.py` - Database Operations
- **Purpose**: Handle all database-related operations
- **Key Classes**: `DatabaseManager`, `SQLQueryBuilder`
- **Responsibilities**:
  - Database connection management
  - SQL query execution
  - SQL validation and security
  - Query pattern generation

```python
from database import DatabaseManager

# Create database manager
db_manager = DatabaseManager(database_url, schema)

# Execute query
results = db_manager.execute_query("SELECT * FROM pet_food LIMIT 10")

# Validate SQL
validation = db_manager.validate_sql_query(sql_query)
```

### 4. `ai_service.py` - AI Integration
- **Purpose**: Handle Google Gemini AI integration
- **Key Classes**: `GeminiAIService`
- **Responsibilities**:
  - AI model configuration
  - SQL query generation from natural language
  - Answer generation from query results
  - Prompt management

```python
from ai_service import GeminiAIService

# Create AI service
ai_service = GeminiAIService(api_key, model_name, schema)

# Generate SQL from natural language
sql_query = ai_service.generate_sql_query("high protein dog food")

# Generate final answer
answer = ai_service.generate_final_answer(user_query, results_df)
```

### 5. `query_processor.py` - Query Processing Logic
- **Purpose**: Orchestrate the complete query processing pipeline
- **Key Classes**: `QueryProcessor`, `BatchQueryProcessor`, `QueryAnalyzer`
- **Responsibilities**:
  - End-to-end query processing
  - Component coordination
  - Batch processing
  - Query analysis and optimization

```python
from query_processor import QueryProcessor, BatchQueryProcessor

# Create query processor
processor = QueryProcessor(database_url, api_key)

# Process single query
result = processor.process_query("show me cat food")

# Batch processing
batch_processor = BatchQueryProcessor(processor)
results = batch_processor.process_batch(["query1", "query2", "query3"])
```

### 6. `display.py` - Display and UI
- **Purpose**: Handle all user interface and output formatting
- **Key Classes**: `DisplayManager`, `InteractiveInterface`, `ResultFormatter`
- **Responsibilities**:
  - Result formatting and display
  - Interactive user interface
  - Output mode management
  - User input/output handling

```python
from display import DisplayManager, InteractiveInterface

# Display results
display_manager = DisplayManager()
display_manager.display_raw_results(dataframe)
display_manager.display_table_view(dataframe)

# Interactive interface
interface = InteractiveInterface()
interface.show_welcome_message()
user_input = interface.get_user_input()
```

### 7. `utils.py` - Utilities and Helpers
- **Purpose**: Provide utility functions and helper classes
- **Key Classes**: `FileManager`, `LoggingSetup`, `InputValidator`, `PerformanceTimer`
- **Responsibilities**:
  - File operations
  - Logging configuration
  - Input validation
  - Performance monitoring
  - Data cleaning utilities

```python
from utils import FileManager, InputValidator, PerformanceTimer

# File operations
file_manager = FileManager()
saved_path = file_manager.save_result_to_file(result_dict)

# Input validation
is_valid, error = InputValidator.validate_user_query(user_input)

# Performance timing
timer = PerformanceTimer()
timer.start()
# ... do work ...
elapsed = timer.stop()
```

### 8. `pet_food_query_system.py` - Main Application
- **Purpose**: Main application interface and entry point
- **Key Classes**: `PetFoodQuerySystem`
- **Responsibilities**:
  - System initialization
  - Component coordination
  - Interactive mode
  - Backward compatibility

```python
from pet_food_query_system import PetFoodQuerySystem

# Initialize system
system = PetFoodQuerySystem()

# Process queries
result = system.query("high protein dog food")

# Run interactive mode
system.run_interactive_mode()
```

## 🔄 Data Flow

```
User Input
    ↓
[InputValidator] → Validation
    ↓
[QueryProcessor] → Coordination
    ↓
[GeminiAIService] → SQL Generation
    ↓
[DatabaseManager] → Query Execution
    ↓
[GeminiAIService] → Answer Generation
    ↓
[DisplayManager] → Result Formatting
    ↓
[FileManager] → Result Saving
    ↓
User Output
```

## ✅ Benefits of Modular Architecture

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Changes in one module don't affect others
- Easier to understand and maintain

### 2. **Testability**
- Individual modules can be tested in isolation
- Mock dependencies for unit testing
- Better test coverage

### 3. **Reusability**
- Components can be reused in other projects
- Mix and match components as needed
- Easy to extend functionality

### 4. **Maintainability**
- Easier to debug and fix issues
- Clear code organization
- Better documentation and readability

### 5. **Scalability**
- Easy to add new features
- Can replace individual components
- Better performance optimization

### 6. **Configuration Management**
- Centralized configuration
- Environment-specific settings
- Easy deployment across environments

## 🚀 Usage Examples

### Basic Usage
```python
from pet_food_query_system import PetFoodQuerySystem

# Simple initialization
system = PetFoodQuerySystem()

# Process a query
result = system.query("high protein dog food")
print(result['answer'])
```

### Advanced Usage
```python
from query_processor import QueryProcessor, BatchQueryProcessor
from display import DisplayManager

# Custom initialization
processor = QueryProcessor(custom_db_url, custom_api_key)

# Batch processing
batch_processor = BatchQueryProcessor(processor)
results = batch_processor.process_batch([
    "high protein dog food",
    "grain-free cat food",
    "senior pet nutrition"
])

# Custom display
display_manager = DisplayManager()
for result in results:
    if not result.error:
        display_manager.display_table_view(result.results)
```

### Configuration Customization
```python
from config import Config

# Override default settings
Config.MAX_RESULTS_LIMIT = 20
Config.QUERY_TIMEOUT = 60

# Validate custom configuration
is_valid, error = Config.validate_config()
```

## 🧪 Testing Strategy

### Unit Testing
- Test each module independently
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Testing
- Test module interactions
- End-to-end query processing
- Database and AI service integration

### Performance Testing
- Query processing speed
- Memory usage optimization
- Concurrent request handling

## 🔧 Development Guidelines

### Adding New Features
1. Identify the appropriate module
2. Follow existing patterns and conventions
3. Add comprehensive documentation
4. Include unit tests
5. Update integration tests

### Modifying Existing Code
1. Understand the module's responsibility
2. Maintain backward compatibility
3. Update related modules if needed
4. Test thoroughly

### Error Handling
- Use appropriate logging levels
- Return meaningful error messages
- Handle edge cases gracefully
- Maintain system stability

## 📈 Future Enhancements

### Potential Improvements
1. **Caching Layer**: Add Redis/memory caching for frequently used queries
2. **API Service**: Create REST API endpoints for remote access
3. **Monitoring**: Add comprehensive monitoring and metrics
4. **Database Abstraction**: Support multiple database backends
5. **Plugin System**: Allow custom extensions and plugins
6. **Async Processing**: Add asynchronous query processing
7. **Web Interface**: Create web-based user interface

### Extension Points
- Custom AI providers (OpenAI, Anthropic, etc.)
- Additional database connectors
- Custom display formats
- New analysis tools
- Export capabilities

## 📚 Migration from Monolithic Version

### Key Changes
1. **Configuration**: Now centralized in `config.py`
2. **Initialization**: Simplified with automatic configuration
3. **Error Handling**: More robust and informative
4. **Logging**: Improved logging throughout the system
5. **Extensibility**: Easy to extend and customize

### Backward Compatibility
The main `PetFoodQuerySystem` class maintains the same interface as the original monolithic version, ensuring existing code continues to work without modifications.

## 🤝 Contributing

When contributing to this modular architecture:
1. Follow the established module responsibilities
2. Maintain clean interfaces between modules
3. Add comprehensive documentation
4. Include appropriate tests
5. Follow Python best practices and PEP 8

---

This modular architecture provides a solid foundation for the Pet Food Query System that is maintainable, testable, and extensible. Each module serves a specific purpose while working together to deliver a robust and user-friendly experience.
