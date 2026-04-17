================================================================================
MLB AI SYSTEM - INTEGRATION GUIDE
================================================================================

A complete Natural Language to SQL system for MLB data queries using:
- FastAPI server with Ollama LLM backend
- Python client with caching and batch processing
- Comprehensive metadata management
- Result export to JSON, CSV, and Markdown

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

CLIENT (mlb_ai_client.py)
    ↓
API REQUESTS
    ↓
FASTAPI SERVER (mlb_fastapi_server.py)
    ├─ LLM: Ollama (llama3)
    ├─ Database: SQL Server
    └─ Metadata: JSON files
    ↓
DATABASE RESPONSES
    ↓
CLIENT RECEIVES & PROCESSES
    ├─ Cache Management
    ├─ Result Export
    └─ Batch Processing

================================================================================
REQUIREMENTS
================================================================================

Python Packages:
  pip install fastapi uvicorn sqlalchemy pyodbc requests langchain-ollama

System Requirements:
  - SQL Server running on KEITH-PERSONAL\SQLEXPRESS
  - Ollama running with llama3 model
  - Python 3.8+

Optional:
  - SQL Server ODBC Driver 18
  - For Windows: pyodbc with SQL Server driver

================================================================================
INSTALLATION STEPS
================================================================================

Step 1: Install Python Dependencies
    pip install fastapi uvicorn sqlalchemy pyodbc requests langchain-ollama

Step 2: Verify Ollama is Running
    ollama serve
    (In another terminal: ollama pull llama3)

Step 3: Create Metadata Folder Structure
    mkdir metadata
    
    Create these JSON files in metadata/:
    - examples.json (example queries)
    - rules.json (SQL generation rules)
    - relationships.json (table join instructions)
    - columns.json (column descriptions)
    - tables.json (table descriptions)

Step 4: Start FastAPI Server
    python mlb_fastapi_server.py
    
    Output should show:
    INFO:     Uvicorn running on http://127.0.0.1:8000
    
    Then in mlb_agent.log:
    - Schema loaded
    - Metadata loaded

Step 5: Run Client
    python mlb_ai_client.py

================================================================================
CLIENT USAGE
================================================================================

BASIC USAGE:
    from mlb_ai_client import MLBQueryClient
    
    client = MLBQueryClient(use_cache=True)
    result = client.ask("TOP 20 schools with most draft picks")
    print(result)

BATCH PROCESSING:
    questions = [
        "Question 1",
        "Question 2",
        "Question 3"
    ]
    
    batch_results = client.ask_batch(questions, verbose=True)
    print(batch_results.summary())

EXPORT RESULTS:
    from mlb_ai_client import ResultExporter
    
    exporter = ResultExporter()
    exporter.export_json(batch_results)
    exporter.export_csv(batch_results)
    exporter.export_markdown(batch_results)

CACHE MANAGEMENT:
    client.cache.clear()  # Clear all cache
    cached = client.cache.get(question)  # Get from cache

UTILITIES:
    # Health check
    is_healthy = client.health_check()
    
    # Get example questions
    examples = client.get_examples()
    
    # Get model info
    model_info = client.get_model_info()

================================================================================
SERVER ENDPOINTS
================================================================================

GET / 
    - Home page with links
    - HTML interface

GET /ask (display form)
    - Interactive query form in browser
    - Can ask questions directly

POST /ask
    - Main query endpoint
    - Input: {"question": "your question"}
    - Output: {
        "sql": "SELECT...",
        "source": "example" or "llm",
        "answer": [[row1], [row2], ...],
        "time_llm": 2.345,
        "time_db": 0.123,
        "time_total": 2.468
      }

GET /examples
    - Returns list of example questions
    - Useful for client-side initialization

GET /model
    - Returns model configuration
    - Shows LLM parameters

GET /docs
    - Auto-generated API documentation (Swagger UI)

================================================================================
METADATA FILES STRUCTURE
================================================================================

metadata/examples.json:
    {
        "examples": [
            {
                "question": "TOP 20 schools with most draft picks",
                "sql": "SELECT TOP 20 school__name, COUNT(*) as pick_count FROM dw.draft GROUP BY school__name ORDER BY pick_count DESC"
            },
            ...
        ]
    }

metadata/rules.json:
    {
        "rules": [
            "Always use SELECT TOP (100) to limit results",
            "Use YEAR(game_date) for date filtering"
        ],
        "common_mistakes": [
            "Using COUNT(*) with GROUP BY without aggregating",
            "Joining on score columns"
        ]
    }

metadata/relationships.json:
    {
        "joins": [
            {
                "from_table": "dw.games",
                "from_column": "teams__home__id",
                "to_table": "dw.teams",
                "to_column": "id",
                "label": "home team"
            }
        ],
        "never_join_on": [
            {
                "pattern": "games.teams__home__score",
                "reason": "Score is not a foreign key, use home team ID"
            }
        ]
    }

metadata/columns.json:
    {
        "dw.games": {
            "game_pk": "Unique game identifier",
            "game_date": "Date of the game",
            "teams__home__score": "Home team's score"
        }
    }

metadata/tables.json:
    {
        "dw.games": "All baseball games with date, teams, score",
        "dw.draft": "Draft picks with player, team, round, pick number"
    }

================================================================================
CONFIGURATION
================================================================================

SERVER CONFIG (mlb_fastapi_server.py):
    
    MAX_RETRIES = 3  # LLM generation attempts
    QUERY_TIMEOUT_SECONDS = 10  # Database query timeout
    AUTO_TOP_LIMIT = 100  # Default result limit
    
    Ollama Config:
        - model="llama3"
        - temperature=0 (deterministic)
        - num_ctx=4096 (context size)
        - num_predict=300 (max tokens)
    
    SQL Server:
        - Server: KEITH-PERSONAL\SQLEXPRESS
        - Database: dlt
        - Auth: Windows Authentication

CLIENT CONFIG (mlb_ai_client.py):
    
    SERVER_URL = "http://127.0.0.1:8000"
    REQUEST_TIMEOUT = 240  # seconds
    CACHE_FOLDER = "query_cache"
    RESULTS_FOLDER = "query_results"
    LOG_FOLDER = "logs"

================================================================================
QUERY FLOW DIAGRAM
================================================================================

1. CLIENT SENDS QUESTION
   ├─ Check cache
   ├─ If cached → Return cached result
   └─ If not cached → Send to server

2. SERVER RECEIVES QUESTION
   ├─ Check example similarity
   ├─ If similar example exists → Return example SQL
   └─ Else → Call LLM

3. LLM GENERATION
   ├─ Enrich question with hints
   ├─ Generate prompt with schema
   ├─ Call Ollama (retry up to 3 times)
   ├─ Extract and validate SQL
   └─ Return SQL to server

4. EXECUTE QUERY
   ├─ Run SQL on database
   ├─ Enforce TOP limit
   ├─ Return results

5. CLIENT PROCESSES RESULT
   ├─ Cache result
   ├─ Format output
   └─ Return to user

================================================================================
EXAMPLE QUESTIONS & EXPECTED PATTERNS
================================================================================

DRAFT QUESTIONS:
    "TOP 20 schools with most draft picks"
    → SELECT TOP 20 school__name, COUNT(*) FROM dw.draft GROUP BY school__name
    
    "Draft picks whose home state is California in 2025"
    → SELECT * FROM dw.draft WHERE home__state='California' AND YEAR(...)=2025

GAMES QUESTIONS:
    "game_pk, home team name, away team name for all games on 2025-06-30"
    → SELECT game_pk, home.name, away.name FROM dw.games g
         JOIN dw.teams home ON g.teams__home__id = home.id
         JOIN dw.teams away ON g.teams__away__id = away.id
         WHERE DATE(g.game_date)='2025-06-30'
    
    "How many games did Giants play Dodgers in 2025"
    → SELECT COUNT(*) FROM dw.games g
         WHERE YEAR(g.game_date)=2025 AND (
            (h.name='Giants' AND a.name='Dodgers') OR
            (h.name='Dodgers' AND a.name='Giants')
         )

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: Server won't start
SOLUTION:
    - Check Ollama is running: ollama serve
    - Check SQL Server connection string
    - Review mlb_agent.log for errors
    - Try: uvicorn mlb_fastapi_server:app --host 127.0.0.1 --port 8000

ISSUE: "Could not connect to server"
SOLUTION:
    - Check server is running on localhost:8000
    - Check firewall allows port 8000
    - Try: curl http://127.0.0.1:8000

ISSUE: LLM keeps timing out
SOLUTION:
    - Check Ollama is responding: curl http://localhost:11434
    - Try: ollama serve (in separate terminal)
    - Increase request_timeout in OllamaLLM config
    - Check system resources (CPU, memory)

ISSUE: SQL injection detected
SOLUTION:
    - Question contains forbidden keywords
    - Try rewording question without: insert, update, delete, drop, etc.
    - Check FORBIDDEN list in mlb_fastapi_server.py

ISSUE: Invalid column error
SOLUTION:
    - Column name may be wrong
    - Check metadata/columns.json for correct names
    - Use metadata/relationships.json for join hints
    - Check table schema in database

ISSUE: No results / Timeout
SOLUTION:
    - Query may be too broad
    - Add more specific filters
    - Check QUERY_TIMEOUT_SECONDS (increase if needed)
    - Check database performance

================================================================================
PERFORMANCE TIPS
================================================================================

1. USE EXAMPLES
   - Add common queries to metadata/examples.json
   - Reduces LLM calls when questions are similar
   - Much faster responses

2. CACHE RESULTS
   - Enable client-side caching (default)
   - Reduces server load
   - Instant results for repeated questions

3. OPTIMIZE METADATA
   - Include clear join instructions
   - List common mistakes to avoid
   - Document complex columns
   - Helps LLM generate correct queries

4. TUNE LLM PARAMETERS
   - temperature=0 for consistency
   - num_predict=300 for longer queries
   - num_ctx=4096 for complex schemas
   - Adjust based on query complexity

5. DATABASE OPTIMIZATION
   - Create indexes on common join columns
   - Use appropriate data types
   - Archive old data if possible
   - Monitor query execution times

================================================================================
RUNNING EXAMPLE
================================================================================

Terminal 1: Start Ollama
    ollama serve

Terminal 2: Start Server
    python mlb_fastapi_server.py
    
    Expected output:
    ✓ Engine and LLM created successfully
    ✓ Connected to database, loading schema...
    ✓ Schema loaded: X tables available
    ✓ Metadata loaded

Terminal 3: Run Client
    python mlb_ai_client.py
    
    Expected output:
    ✓ Server is running
    Model: llama3
    [1/7] Processing...
    ✓ SUCCESS: 20 rows
    [2/7] Processing...
    ✓ SUCCESS: 5 rows
    ...
    
    BATCH PROCESSING SUMMARY
    Total Queries: 7
    Successful: 7 (100%)
    Failed: 0
    ...
    ✓ Results exported to: query_results

================================================================================
FILE STRUCTURE
================================================================================

project/
├── mlb_fastapi_server.py       # FastAPI server
├── mlb_ai_client.py            # Client with caching
├── metadata/                    # LLM guidance files
│   ├── examples.json
│   ├── rules.json
│   ├── relationships.json
│   ├── columns.json
│   └── tables.json
├── query_cache/                 # Client-side cache (auto-created)
├── query_results/               # Exported results (auto-created)
│   ├── results_*.json
│   ├── results_*.csv
│   └── results_*.md
├── logs/                        # Log files (auto-created)
│   ├── mlb_agent.log           # Server logs
│   └── mlb_client_*.log        # Client logs
└── README.md

================================================================================
NEXT STEPS
================================================================================

1. CREATE METADATA FILES
   - Fill in examples.json with common queries
   - Add rules and relationships
   - Document important columns

2. TEST MANUALLY
   - Visit http://127.0.0.1:8000/ask
   - Try a few questions
   - Check results and SQL

3. RUN BATCH PROCESSING
   - Use provided test questions
   - Review generated SQL
   - Export results

4. OPTIMIZE
   - Add more examples
   - Tune LLM parameters
   - Monitor performance

5. INTEGRATE
   - Use client in your application
   - Handle errors appropriately
   - Cache results
   - Export as needed

================================================================================
SUPPORT & DEBUGGING
================================================================================

LOGS TO CHECK:
    - mlb_agent.log (server)
    - logs/mlb_client_*.log (client)
    - query_cache/* (cached queries)

API DOCUMENTATION:
    - Open http://127.0.0.1:8000/docs
    - Swagger UI with all endpoints
    - Try requests directly

TESTING:
    - Manual questions via web interface
    - Batch processing with test_questions
    - Cache performance checks
    - Export format validation

DEBUGGING:
    1. Check logs first
    2. Test health endpoint
    3. Try simpler question
    4. Review generated SQL
    5. Check metadata files
    6. Verify database connection

================================================================================
VERSION
================================================================================

MLB AI System v1.0
Created: 2024
FastAPI + Ollama + SQL Server Integration
Production Ready

================================================================================
