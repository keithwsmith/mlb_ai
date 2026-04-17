"""
MLB AI Query Client
Integrates with FastAPI backend for natural language SQL generation and execution
Supports batch processing, caching, detailed performance metrics, and test mode
Reads questions from questions.json or examples.json (test mode)
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import io
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

SERVER_URL      = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 600           # seconds — allows for 3 LLM retries
CACHE_FOLDER    = "query_cache"
RESULTS_FOLDER  = "query_results"
LOG_FOLDER      = "logs"
QUESTIONS_FILE  = "questions.json"   # default questions file
EXAMPLES_FILE   = "examples.json"    # examples / test file


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging to both file and console."""
    Path(LOG_FOLDER).mkdir(exist_ok=True)

    log_file = os.path.join(
        LOG_FOLDER,
        f"mlb_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Force UTF-8 on the console stream so Unicode chars don't crash cp1252
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(utf8_stdout)
        ]
    )

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# QUESTIONS LOADER
# ============================================================================

def load_questions(filepath: str = QUESTIONS_FILE) -> List[str]:
    """
    Load questions from a JSON file.

    Supports two formats:

    Format 1 — simple array:
        ["question 1", "question 2", ...]

    Format 2 — object with questions key:
        {
          "questions": ["question 1", "question 2", ...]
        }

    Format 3 — object with category groups:
        {
          "draft":   ["question 1", "question 2"],
          "games":   ["question 3", "question 4"],
          "hitters": ["question 5"]
        }
    """
    if not os.path.exists(filepath):
        logger.error(f"Questions file not found: {filepath}")
        print(f"ERROR: Questions file not found: {filepath}")
        print(f"       Create '{filepath}' with a list of questions to run.")
        return []

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Format 1 — bare list
        if isinstance(data, list):
            questions = [q for q in data if isinstance(q, str) and q.strip()]
            logger.info(f"Loaded {len(questions)} questions from {filepath} (list format)")
            return questions

        # Format 2 — {"questions": [...]}
        if isinstance(data, dict) and "questions" in data:
            questions = [q for q in data["questions"]
                         if isinstance(q, str) and q.strip()]
            logger.info(f"Loaded {len(questions)} questions from {filepath} (questions key)")
            return questions

        # Format 3 — {"category": [...], "category2": [...]}
        if isinstance(data, dict):
            questions = []
            for category, items in data.items():
                if isinstance(items, list):
                    cat_questions = [q for q in items
                                     if isinstance(q, str) and q.strip()]
                    logger.info(f"  Category '{category}': {len(cat_questions)} questions")
                    questions.extend(cat_questions)
            logger.info(f"Loaded {len(questions)} total questions from {filepath} (category format)")
            return questions

        logger.error(f"Unrecognized format in {filepath}")
        return []

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        print(f"ERROR: Invalid JSON in {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Could not load {filepath}: {e}")
        print(f"ERROR: Could not load {filepath}: {e}")
        return []


# ============================================================================
# EXAMPLES LOADER  (for test mode)
# ============================================================================

def load_examples(filepath: str = EXAMPLES_FILE) -> List[Dict[str, str]]:
    """
    Load question+SQL pairs from examples.json.

    Supports two formats:

    Format 1 — bare list of {question, sql} objects:
        [{"question": "...", "sql": "..."}, ...]

    Format 2 — object with examples key:
        {"examples": [{"question": "...", "sql": "..."}, ...]}
    """
    if not os.path.exists(filepath):
        logger.error(f"Examples file not found: {filepath}")
        print(f"ERROR: Examples file not found: {filepath}")
        return []

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap {"examples": [...]} wrapper if present
        if isinstance(data, dict) and "examples" in data:
            data = data["examples"]

        if not isinstance(data, list):
            logger.error(f"Expected a list of examples in {filepath}")
            return []

        examples = [
            e for e in data
            if isinstance(e, dict)
            and isinstance(e.get("question"), str) and e["question"].strip()
            and isinstance(e.get("sql"), str) and e["sql"].strip()
        ]

        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        print(f"ERROR: Invalid JSON in {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Could not load {filepath}: {e}")
        print(f"ERROR: Could not load {filepath}: {e}")
        return []


# ============================================================================
# SQL NORMALISATION  (for test comparison)
# ============================================================================

def normalize_sql(sql: str) -> str:
    """
    Normalise SQL for comparison:
      - collapse whitespace
      - uppercase keywords / identifiers
      - strip trailing semicolons
    """
    import re
    sql = sql.strip().rstrip(";").upper()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def sql_match(expected: str, actual: str) -> bool:
    """Return True when normalised SQL strings are identical."""
    return normalize_sql(expected) == normalize_sql(actual)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QueryResult:
    """Represents a single query result."""
    question:   str
    sql:        str
    rows:       List[List]
    source:     str
    time_llm:   float
    time_db:    float
    time_total: float
    error:      Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows) if self.rows else 0

    def to_dict(self) -> Dict:
        return {
            'question':   self.question,
            'sql':        self.sql,
            'rows':       self.rows,
            'source':     self.source,
            'time_llm':   self.time_llm,
            'time_db':    self.time_db,
            'time_total': self.time_total,
            'error':      self.error,
            'success':    self.success,
            'row_count':  self.row_count
        }

    def __str__(self) -> str:
        status = "[SUCCESS]" if self.success else "[ERROR]"
        output = f"\n{status}\n"
        output += f"Question: {self.question}\n"

        if self.error:
            output += f"Error: {self.error}\n"
        else:
            output += f"SQL: {self.sql}\n"
            output += f"Rows: {self.row_count}\n"
            if self.rows:
                output += "Results:\n"
                for i, row in enumerate(self.rows[:10], 1):   # cap display at 10
                    output += f"  {i}. {row}\n"
                if self.row_count > 10:
                    output += f"  ... ({self.row_count - 10} more rows)\n"

        output += (
            f"Source: {self.source} | "
            f"LLM: {self.time_llm}s | "
            f"DB: {self.time_db}s | "
            f"Total: {self.time_total}s\n"
        )
        return output


@dataclass
class TestResult:
    """Represents the outcome of a single example test."""
    question:     str
    expected_sql: str
    actual_sql:   str
    sql_match:    bool
    rows:         List[List]
    source:       str
    time_llm:     float
    time_db:      float
    time_total:   float
    error:        Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.sql_match

    @property
    def row_count(self) -> int:
        return len(self.rows) if self.rows else 0

    def to_dict(self) -> Dict:
        return {
            'question':     self.question,
            'expected_sql': self.expected_sql,
            'actual_sql':   self.actual_sql,
            'sql_match':    self.sql_match,
            'passed':       self.passed,
            'rows':         self.rows,
            'source':       self.source,
            'time_llm':     self.time_llm,
            'time_db':      self.time_db,
            'time_total':   self.time_total,
            'error':        self.error,
            'row_count':    self.row_count,
        }

    def __str__(self) -> str:
        icon   = "[PASS]" if self.passed else "[FAIL]"
        output = f"\n{icon}\n"
        output += f"Question : {self.question}\n"

        if self.error:
            output += f"Error    : {self.error}\n"
        else:
            match_label = "MATCH" if self.sql_match else "MISMATCH"
            output += f"SQL      : [{match_label}]\n"
            if not self.sql_match:
                output += f"  Expected : {self.expected_sql}\n"
                output += f"  Actual   : {self.actual_sql}\n"
            else:
                output += f"  {self.actual_sql}\n"
            output += f"Rows     : {self.row_count}\n"

        output += (
            f"Source: {self.source} | "
            f"LLM: {self.time_llm}s | "
            f"DB: {self.time_db}s | "
            f"Total: {self.time_total}s\n"
        )
        return output


@dataclass
class BatchResults:
    """Represents batch processing results."""
    results:    List[QueryResult]
    total_time: float
    successful: int
    failed:     int

    @property
    def success_rate(self) -> float:
        total = len(self.results)
        return (self.successful / total * 100) if total > 0 else 0

    def summary(self) -> str:
        """Get summary statistics — ASCII only to avoid cp1252 encoding errors."""
        total_rows = sum(r.row_count for r in self.results)
        avg_time   = (sum(r.time_total for r in self.results) / len(self.results)
                      if self.results else 0)

        lines = [
            "",
            "==========================================",
            "       BATCH PROCESSING SUMMARY          ",
            "==========================================",
            "",
            f"  Total Queries  : {len(self.results)}",
            f"  Successful     : {self.successful} ({self.success_rate:.1f}%)",
            f"  Failed         : {self.failed}",
            "",
            f"  Total Rows     : {total_rows}",
            f"  Avg Query Time : {avg_time:.3f}s",
            f"  Total Time     : {self.total_time:.2f}s",
            "",
            "==========================================",
            "",
        ]

        if self.failed > 0:
            lines.append("  FAILED QUERIES:")
            for r in self.results:
                if not r.success:
                    lines.append(f"    - {r.question[:80]}")
                    lines.append(f"      Error: {r.error[:120]}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class TestBatchResults:
    """Represents batch test results from examples.json."""
    results:    List[TestResult]
    total_time: float
    passed:     int
    failed:     int
    errors:     int

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0

    @property
    def sql_match_count(self) -> int:
        return sum(1 for r in self.results if r.sql_match and not r.error)

    def summary(self) -> str:
        avg_time = (
            sum(r.time_total for r in self.results) / self.total
            if self.total > 0 else 0
        )

        lines = [
            "",
            "==========================================",
            "         TEST RUN SUMMARY                ",
            "==========================================",
            "",
            f"  Total Tests    : {self.total}",
            f"  Passed         : {self.passed} ({self.pass_rate:.1f}%)",
            f"  Failed (SQL)   : {self.failed}",
            f"  Errors         : {self.errors}",
            "",
            f"  SQL Match Rate : {self.sql_match_count}/{self.total - self.errors}",
            f"  Avg Query Time : {avg_time:.3f}s",
            f"  Total Time     : {self.total_time:.2f}s",
            "",
            "==========================================",
            "",
        ]

        if self.failed > 0 or self.errors > 0:
            lines.append("  FAILED / ERROR TESTS:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"    - {r.question[:80]}")
                    if r.error:
                        lines.append(f"      Error: {r.error[:120]}")
                    elif not r.sql_match:
                        lines.append(f"      Expected : {r.expected_sql[:100]}")
                        lines.append(f"      Actual   : {r.actual_sql[:100]}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class QueryCache:
    """Simple file-based cache for query results."""

    def __init__(self, cache_folder: str = CACHE_FOLDER):
        self.cache_folder = cache_folder
        Path(cache_folder).mkdir(exist_ok=True)
        logger.info(f"Cache initialized: {cache_folder}")

    def _get_cache_key(self, question: str) -> str:
        import hashlib
        return hashlib.md5(question.encode()).hexdigest()

    def _get_cache_file(self, question: str) -> str:
        key = self._get_cache_key(question)
        return os.path.join(self.cache_folder, f"{key}.json")

    def get(self, question: str) -> Optional[QueryResult]:
        cache_file = self._get_cache_file(question)
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, encoding='utf-8') as f:
                data = json.load(f)
                # Strip computed properties that are not constructor args
                data.pop('success', None)
                data.pop('row_count', None)
                result = QueryResult(**data)
            logger.debug(f"Cache HIT: {question[:60]}...")
            return result
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def put(self, result: QueryResult):
        cache_file = self._get_cache_file(result.question)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.debug(f"Cache STORE: {result.question[:60]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        import shutil
        try:
            shutil.rmtree(self.cache_folder)
            Path(self.cache_folder).mkdir(exist_ok=True)
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def size(self) -> int:
        """Return number of cached entries."""
        try:
            return len([f for f in os.listdir(self.cache_folder)
                        if f.endswith('.json')])
        except Exception:
            return 0


# ============================================================================
# MLB AI CLIENT
# ============================================================================

class MLBQueryClient:
    """Client for MLB AI Query API."""

    def __init__(self, server_url: str = SERVER_URL, use_cache: bool = True):
        self.server_url = server_url.rstrip('/')
        self.cache      = QueryCache() if use_cache else None
        self.use_cache  = use_cache
        logger.info(
            f"Client initialized: {server_url} "
            f"(cache={'enabled' if use_cache else 'disabled'})"
        )

    def ask(self, question: str) -> QueryResult:
        logger.info(f"QUESTION: {question}")

        if self.use_cache:
            cached = self.cache.get(question)
            if cached:
                logger.info(f"Cache hit for: {question[:60]}")
                return cached

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/ask",
                json={"question": question},
                timeout=REQUEST_TIMEOUT
            )

            data       = response.json()
            total_time = time.time() - start_time

            if "error" in data:
                result = QueryResult(
                    question=question, sql="", rows=[],
                    source="error",
                    time_llm=0, time_db=0, time_total=total_time,
                    error=data["error"]
                )
                logger.error(f"Query error: {data['error']}")
            else:
                result = QueryResult(
                    question=question,
                    sql=data.get("sql", ""),
                    rows=data.get("answer", []),
                    source=data.get("source", "unknown"),
                    time_llm=data.get("time_llm", 0),
                    time_db=data.get("time_db", 0),
                    time_total=data.get("time_total", total_time)
                )
                logger.info(
                    f"Query successful: {result.row_count} rows "
                    f"(source={result.source})"
                )

            if self.use_cache and result.success:
                self.cache.put(result)

            return result

        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return QueryResult(
                question=question, sql="", rows=[], source="error",
                time_llm=0, time_db=0, time_total=REQUEST_TIMEOUT,
                error=f"Request timeout after {REQUEST_TIMEOUT}s"
            )
        except requests.exceptions.ConnectionError:
            logger.error("Connection error")
            return QueryResult(
                question=question, sql="", rows=[], source="error",
                time_llm=0, time_db=0, time_total=0,
                error=f"Could not connect to server at {self.server_url}"
            )
        except Exception as e:
            logger.error(f"Client error: {e}")
            return QueryResult(
                question=question, sql="", rows=[], source="error",
                time_llm=0, time_db=0, time_total=0,
                error=str(e)
            )

    def ask_batch(
        self,
        questions: List[str],
        verbose: bool = True
    ) -> BatchResults:
        logger.info(f"Starting batch processing: {len(questions)} questions")

        results    = []
        start_time = time.time()

        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[{i}/{len(questions)}] {question[:80]}")
                print("-" * 60)

            result = self.ask(question)
            results.append(result)

            if verbose:
                print(result)

        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed     = len(results) - successful

        batch_results = BatchResults(
            results=results,
            total_time=total_time,
            successful=successful,
            failed=failed
        )

        logger.info(batch_results.summary())
        return batch_results

    # ------------------------------------------------------------------
    # TEST MODE  — run examples.json and compare generated SQL
    # ------------------------------------------------------------------

    def run_tests(
        self,
        examples: List[Dict[str, str]],
        verbose:  bool = True
    ) -> TestBatchResults:
        """
        Run each example from examples.json as a test case.

        For every {question, sql} pair:
          1. Send the question to /ask
          2. Compare the returned SQL to the expected SQL (normalised)
          3. Record PASS / FAIL
        """
        logger.info(f"Starting test run: {len(examples)} examples")

        results    = []
        start_time = time.time()

        for i, example in enumerate(examples, 1):
            question     = example["question"]
            expected_sql = example["sql"]

            if verbose:
                print(f"\n[{i}/{len(examples)}] {question[:80]}")
                print("-" * 60)

            qr = self.ask(question)

            matched = sql_match(expected_sql, qr.sql) if qr.success else False

            tr = TestResult(
                question     = question,
                expected_sql = expected_sql,
                actual_sql   = qr.sql,
                sql_match    = matched,
                rows         = qr.rows,
                source       = qr.source,
                time_llm     = qr.time_llm,
                time_db      = qr.time_db,
                time_total   = qr.time_total,
                error        = qr.error,
            )
            results.append(tr)

            if verbose:
                print(tr)

        total_time = time.time() - start_time
        passed     = sum(1 for r in results if r.passed)
        errors     = sum(1 for r in results if r.error)
        failed     = len(results) - passed - errors

        test_results = TestBatchResults(
            results    = results,
            total_time = total_time,
            passed     = passed,
            failed     = failed,
            errors     = errors,
        )

        logger.info(test_results.summary())
        return test_results

    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            response   = requests.get(f"{self.server_url}/", timeout=5)
            is_healthy = response.status_code == 200
            logger.info(f"Health check: {'OK' if is_healthy else 'FAILED'}")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_examples(self) -> List[str]:
        try:
            response = requests.get(f"{self.server_url}/examples", timeout=5)
            data     = response.json()
            logger.info(f"Retrieved {data['count']} examples from server")
            return data.get("questions", [])
        except Exception as e:
            logger.error(f"Could not retrieve examples: {e}")
            return []

    def get_model_info(self) -> Dict:
        try:
            response = requests.get(f"{self.server_url}/model", timeout=5)
            data     = response.json()
            logger.info(f"Model: {data.get('model')}")
            return data
        except Exception as e:
            logger.error(f"Could not retrieve model info: {e}")
            return {}


# ============================================================================
# RESULT EXPORT
# ============================================================================

class ResultExporter:
    """Export results to JSON, CSV, and Markdown formats."""

    @staticmethod
    def export_json(results: BatchResults, filename: str = None) -> str:
        if not filename:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        data = {
            'timestamp':    datetime.now().isoformat(),
            'total_queries': len(results.results),
            'successful':   results.successful,
            'failed':       results.failed,
            'success_rate': results.success_rate,
            'total_time':   results.total_time,
            'results':      [r.to_dict() for r in results.results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results exported to {filepath}")
        return filepath

    @staticmethod
    def export_csv(results: BatchResults, filename: str = None) -> str:
        import csv

        if not filename:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Question', 'Success', 'Row Count', 'Source',
                'LLM Time (s)', 'DB Time (s)', 'Total Time (s)', 'Error'
            ])
            for result in results.results:
                writer.writerow([
                    result.question,
                    'Yes' if result.success else 'No',
                    result.row_count,
                    result.source,
                    result.time_llm,
                    result.time_db,
                    result.time_total,
                    result.error or ''
                ])

        logger.info(f"Results exported to {filepath}")
        return filepath

    @staticmethod
    def export_markdown(results: BatchResults, filename: str = None) -> str:
        if not filename:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# MLB AI Query Results\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total Queries: {len(results.results)}\n")
            f.write(f"- Successful: {results.successful}\n")
            f.write(f"- Failed: {results.failed}\n")
            f.write(f"- Success Rate: {results.success_rate:.1f}%\n")
            f.write(f"- Total Time: {results.total_time:.2f}s\n\n")

            f.write("## Results\n\n")
            for i, result in enumerate(results.results, 1):
                f.write(f"### Query {i}\n\n")
                f.write(f"**Question**: {result.question}\n\n")
                f.write(
                    f"**Status**: {'SUCCESS' if result.success else 'FAILED'}\n\n")

                if result.error:
                    f.write(f"**Error**: {result.error}\n\n")
                else:
                    f.write(f"**SQL**:\n```sql\n{result.sql}\n```\n\n")
                    f.write(f"**Rows**: {result.row_count}\n\n")
                    if result.rows and len(result.rows) <= 10:
                        f.write("**Results**:\n```\n")
                        for row in result.rows:
                            f.write(f"{row}\n")
                        f.write("```\n\n")

                f.write(
                    f"**Timing**: "
                    f"LLM={result.time_llm}s | "
                    f"DB={result.time_db}s | "
                    f"Total={result.time_total}s\n"
                )
                f.write(f"**Source**: {result.source}\n\n")
                f.write("---\n\n")

        logger.info(f"Results exported to {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Test-mode export methods
    # ------------------------------------------------------------------

    @staticmethod
    def export_tests_json(results: TestBatchResults, filename: str = None) -> str:
        if not filename:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        data = {
            'timestamp':  datetime.now().isoformat(),
            'total':      results.total,
            'passed':     results.passed,
            'failed':     results.failed,
            'errors':     results.errors,
            'pass_rate':  results.pass_rate,
            'total_time': results.total_time,
            'results':    [r.to_dict() for r in results.results],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Test results exported to {filepath}")
        return filepath

    @staticmethod
    def export_tests_csv(results: TestBatchResults, filename: str = None) -> str:
        import csv

        if not filename:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Question', 'Passed', 'SQL Match', 'Row Count', 'Source',
                'LLM Time (s)', 'DB Time (s)', 'Total Time (s)',
                'Expected SQL', 'Actual SQL', 'Error'
            ])
            for r in results.results:
                writer.writerow([
                    r.question,
                    'PASS' if r.passed else 'FAIL',
                    'YES'  if r.sql_match else 'NO',
                    r.row_count,
                    r.source,
                    r.time_llm,
                    r.time_db,
                    r.time_total,
                    r.expected_sql,
                    r.actual_sql,
                    r.error or '',
                ])

        logger.info(f"Test results exported to {filepath}")
        return filepath

    @staticmethod
    def export_tests_markdown(results: TestBatchResults, filename: str = None) -> str:
        if not filename:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        filepath = os.path.join(RESULTS_FOLDER, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# MLB AI Query — Test Results\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Total Tests | {results.total} |\n")
            f.write(f"| Passed | {results.passed} |\n")
            f.write(f"| Failed (SQL mismatch) | {results.failed} |\n")
            f.write(f"| Errors | {results.errors} |\n")
            f.write(f"| Pass Rate | {results.pass_rate:.1f}% |\n")
            f.write(f"| Total Time | {results.total_time:.2f}s |\n\n")

            f.write("## Test Cases\n\n")
            for i, r in enumerate(results.results, 1):
                status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
                f.write(f"### Test {i} — {status}\n\n")
                f.write(f"**Question**: {r.question}\n\n")

                if r.error:
                    f.write(f"**Error**: {r.error}\n\n")
                else:
                    match_label = "MATCH" if r.sql_match else "MISMATCH"
                    f.write(f"**SQL**: {match_label}\n\n")
                    if not r.sql_match:
                        f.write(f"Expected:\n```sql\n{r.expected_sql}\n```\n\n")
                        f.write(f"Actual:\n```sql\n{r.actual_sql}\n```\n\n")
                    else:
                        f.write(f"```sql\n{r.actual_sql}\n```\n\n")
                    f.write(f"**Rows**: {r.row_count}\n\n")

                f.write(
                    f"**Timing**: "
                    f"LLM={r.time_llm}s | "
                    f"DB={r.time_db}s | "
                    f"Total={r.time_total}s\n"
                )
                f.write(f"**Source**: {r.source}\n\n")
                f.write("---\n\n")

        logger.info(f"Test results exported to {filepath}")
        return filepath


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MLB AI Query Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py                             # run questions.json
  python client.py -f my_questions.json        # run a specific questions file
  python client.py --test                      # run examples.json as tests
  python client.py --test -f examples.json     # run a specific examples file as tests
  python client.py -f questions.json --no-cache
  python client.py --list-examples             # show examples loaded on server
        """
    )
    parser.add_argument(
        "-f", "--file",
        default=None,
        help=(
            f"Input file to run. "
            f"Defaults to '{EXAMPLES_FILE}' in --test mode, "
            f"'{QUESTIONS_FILE}' otherwise."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            f"Test mode: read {{question, sql}} pairs from examples.json "
            f"and compare generated SQL to expected SQL"
        )
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching"
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List examples loaded on the server and exit"
    )
    parser.add_argument(
        "--server",
        default=SERVER_URL,
        help=f"Server URL (default: {SERVER_URL})"
    )

    args = parser.parse_args()

    # Resolve default file based on mode
    if args.file is None:
        args.file = EXAMPLES_FILE if args.test else QUESTIONS_FILE

    print("=" * 70)
    print("MLB AI QUERY CLIENT" + (" — TEST MODE" if args.test else ""))
    print("=" * 70)

    client = MLBQueryClient(
        server_url=args.server,
        use_cache=not args.no_cache
    )

    # Health check
    if not client.health_check():
        print("\nERROR: Server is not running!")
        print(f"  Start the server: python app.py")
        print(f"  Server URL: {args.server}")
        return

    print("\nServer is running")

    # Model info
    model_info = client.get_model_info()
    if model_info:
        print(f"Model    : {model_info.get('model')}")
        print(f"Context  : {model_info.get('num_ctx')}")
        print(f"Predict  : {model_info.get('num_predict')}")

    # Cache info
    if not args.no_cache:
        cache_size = client.cache.size()
        print(f"Cache    : {cache_size} cached entries in '{CACHE_FOLDER}/'")

    # --list-examples mode
    if args.list_examples:
        print("\nExamples loaded on server:")
        print("-" * 60)
        examples = client.get_examples()
        if examples:
            for i, q in enumerate(examples, 1):
                print(f"  {i:3}. {q}")
        else:
            print("  (none)")
        return

    # ----------------------------------------------------------------
    # TEST MODE
    # ----------------------------------------------------------------
    if args.test:
        print(f"\nLoading examples from: {args.file}")
        examples = load_examples(args.file)

        if not examples:
            print(f"\nNo examples found in '{args.file}'.")
            print("Ensure the file contains [{\"question\": \"...\", \"sql\": \"...\"}] pairs.")
            return

        print(f"Found {len(examples)} example(s)\n")

        test_results = client.run_tests(examples, verbose=True)

        print(test_results.summary())

        Path(RESULTS_FOLDER).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        json_path = ResultExporter.export_tests_json(
            test_results, f"test_results_{timestamp}.json")
        csv_path  = ResultExporter.export_tests_csv(
            test_results, f"test_results_{timestamp}.csv")
        md_path   = ResultExporter.export_tests_markdown(
            test_results, f"test_results_{timestamp}.md")

        print(f"Test results exported to '{RESULTS_FOLDER}/':")
        print(f"  {os.path.basename(json_path)}")
        print(f"  {os.path.basename(csv_path)}")
        print(f"  {os.path.basename(md_path)}")
        return

    # ----------------------------------------------------------------
    # NORMAL / BATCH MODE
    # ----------------------------------------------------------------
    print(f"\nLoading questions from: {args.file}")
    questions = load_questions(args.file)

    if not questions:
        print(f"\nNo questions found in '{args.file}'.")
        print("Create the file with one of these formats:\n")
        print('  Simple list:')
        print('  ["question 1", "question 2"]\n')
        print('  Object with questions key:')
        print('  {"questions": ["question 1", "question 2"]}\n')
        print('  Grouped by category:')
        print('  {"draft": ["question 1"], "games": ["question 2"]}')
        return

    print(f"Found {len(questions)} question(s)\n")

    # Run batch
    batch_results = client.ask_batch(questions, verbose=True)

    # Print summary
    print(batch_results.summary())

    # Export results
    Path(RESULTS_FOLDER).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = ResultExporter.export_json(
        batch_results, f"results_{timestamp}.json")
    csv_path  = ResultExporter.export_csv(
        batch_results, f"results_{timestamp}.csv")
    md_path   = ResultExporter.export_markdown(
        batch_results, f"results_{timestamp}.md")

    print(f"Results exported to '{RESULTS_FOLDER}/':")
    print(f"  {os.path.basename(json_path)}")
    print(f"  {os.path.basename(csv_path)}")
    print(f"  {os.path.basename(md_path)}")


if __name__ == "__main__":
    main()
