"""
MLB AI FastAPI Server
Serves the NLP query backend with Ollama sqlcoder integration
Includes Reflection pattern: /reflect, /reflect/query, /reflect/tables, /reflect/examples

CHANGES vs previous version
────────────────────────────
1. Lifespan: loud startup warning + print if examples loaded 0 entries (catches JSON parse failures)
2. load_metadata: prints JSON parse errors to stdout so they're visible at startup
3. GAME_PROMPT_NOTES: heavily expanded with flat-column examples, banned 'is_home_run',
   added correct 'events = home_run' usage, added GROUP BY + ORDER BY MIN() example,
   added flat-column day_night and doubleheader count examples
4. enforce_top: no longer injects TOP into COUNT(*) queries without GROUP BY
5. validate_groupby: added MIN/MAX to the aggregate check
6. is_game_question regex: expanded to cover day_night, doubleheader, series_description, game_pk
7. Error hint: added 'is_home_run' to the invalid column retry message
8. Year range support: has_year_range check bypasses example cache for BETWEEN queries
9. enrich_question: table-aware year range hints (d.year, h.season, p.season, g.season, r.season)
10. filter_relevant_tables: draft queries no longer include ALWAYS_INCLUDE_TABLES
11. validate_draft_no_join: explicit validator rejects any JOIN dw.teams on draft queries
12. validate_draft_no_join fires BEFORE validate_columns
13. Retry hints: dedicated draft-join branch with embedded correct SQL example
14. ROSTER_PROMPT_NOTES: added roster prompt block + is_roster routing in generate_sql
15. roster_hallucination_keywords: targeted retry hint for hallucinated roster tables
16. enrich_question: added is_roster branch with r.season hint (prevents games hint bleeding in)
17. POSTGRES_PATTERNS: added to_number() detection
18. base_prompt FORBIDDEN list: added to_number
19. fix_ambiguous_columns: qualifies bare roster column refs with r. alias (fixes ambiguous 'season')
20. sql_contains_year: word-boundary year check replaces naive substring match in example cache
21. sql_matches_question_context: validates cached SQL years AND team name before reusing example
    -- prevents cross-team cache hits and wrong-year reuse (e.g. 2025-2026 question getting
       2024-2025 cached SQL returned as a match)

OPTIMIZATION CHANGES (v2)
──────────────────────────
OPT-1. Model upgrade: bump OLLAMA_MODEL default to sqlcoder:15b
OPT-2. Semantic example cache: sentence-transformers replaces SequenceMatcher
        -- all-MiniLM-L6-v2 embeddings built at startup; cosine similarity lookup
        -- threshold lowered to 0.88 (semantic similarity is stricter than string ratio)
        -- graceful fallback to SequenceMatcher if sentence-transformers not installed
OPT-3. Single domain-block prompt dispatch: get_query_type() + PROMPT_BLOCKS dict
        -- replaces six parallel is_X_question booleans and empty block variables
        -- cuts prompt length ~40%, reduces 7b/15b context confusion
OPT-4. Column relevance filtering: filter_columns_for_question()
        -- keeps always-keep columns + keyword-matched columns, respects per-table cap
        -- reduces schema noise sent to LLM without hiding critical columns
OPT-5. Bug fix: fix_ambiguous_columns now runs BEFORE validate_columns
        -- prevents false validation failures on bare roster column refs
OPT-6. Auto-promote LLM results to examples: maybe_save_example()
        -- successful LLM queries with 1..99 rows are appended to examples.json
        -- semantic embeddings hot-reloaded after each new save

BUG FIXES (v3)
──────────────
FIX-1. get_query_type: added missing final `return "game"` -- previously returned None
        for most questions, causing wrong domain block and schema selection
FIX-2. filter_relevant_tables: added is_game_record block BEFORE keyword fallback loop
        -- "record against", "opponent", "head-to-head", "since YYYY" now correctly
           route to GAME_TABLES (dw.games, dw.teams, dw.venues) instead of
           dw.team_season_features via keyword match on "team"
FIX-3. GAME_PROMPT_NOTES: added teams__home__id / teams__away__id to BANNED columns
        -- prevents LLM from inventing JOIN dw.teams on fake ID columns
FIX-4. warmup_model: replaced llm.invoke("hi") with direct Ollama /api/generate call
        using num_predict=1 and num_ctx=128 -- avoids full context allocation on warmup

ORDERING FIX (v4)
─────────────────
ORD-1. ORDERING_RULES block: explicit ASC/DESC rules injected into every prompt
        -- "worst/lowest/fewest" → ORDER BY ASC
        -- "best/highest/most"   → ORDER BY DESC
ORD-2. validate_ordering_direction: post-generation validator catches direction mismatches
        -- fires after validate_cast_has_type in the retry loop
ORD-3. Retry hint: dedicated ordering error branch with corrected SQL example
        -- placed before generic else branch in retry hint dispatch

JC FIX (v5)
───────────
JC-1. DRAFT_PROMPT_EXAMPLE: added junior college example using dw.school_type_lookup JOIN
      and fallback LIKE pattern when lookup unavailable
JC-2. validate_school_type_filter: new validator fires when question mentions
      junior college / juco / community college but SQL has no school_type filter
JC-3. Retry hint: dedicated juco_filter branch with correct JOIN example
JC-4. DRAFT_PROMPT_EXAMPLE: school__name column note updated to reference school_type_lookup

TIMEOUT + ROUTING FIXES (v6)
─────────────────────────────
TO-1. Timeout middleware: 150s hard cap on all requests -- returns 504 instead of hanging
TO-2. Opening Day routing: \bopening.day\b removed from is_season, dedicated is_opening_day
      block routes to GAME_TABLES instead of SEASON_TABLES
TO-3. Pitch core routing: is_pitch_core block routes hits/pitch_type/batter_side questions
      to PITCH_CORE_TABLES before keyword fallback
TO-4. PITCH_CORE_PROMPT: new domain block for dw.pitch_core queries
      -- defines p.events values for hits, p.stand, p.p_throws, p.pitch_type
      -- added pitch_core to get_query_type() and PROMPT_BLOCKS
TO-5. Hitter routing: added hitter.season / season.stats keywords to is_hitter regex
      in both get_query_type() and filter_relevant_tables()
TO-6. Hitter team hint: enrich_question now appends team name hint for hitter queries
      -- uses h.team_name with fallback note
TO-7. maybe_save_example: fixed dead code bug -- JC guard was unreachable due to
      indentation under early return; removed duplicate row count check
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from contextlib import asynccontextmanager
from langchain_ollama import OllamaLLM
import urllib.parse
import logging
import time
import re
import os
import json
import asyncio
from difflib import SequenceMatcher
import httpx

os.environ["PYTHONNOUSERSITE"] = "1"


# ============================
# OPTIONAL: sentence-transformers
# ============================
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# ============================
# CONFIG
# ============================

MAX_RETRIES = 3
QUERY_TIMEOUT_SECONDS = 10
AUTO_TOP_LIMIT = 100

OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "sqlcoder:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.0.0.54:11434")

EXCLUDE_TABLES = [
    "dw.audit_log",
    "dw.etl_log",
    "dw._dlt_pipeline_state",
    "dw._dlt_version",
    "dw.team_season_features",
    "dw.etltablelog",
    "dw.play_events"
]

DRAFT_TABLES        = ["dw.draft", "dw.school_type_lookup"]
GAME_TABLES         = ["dw.games", "dw.teams", "dw.venues"]
HITTER_TABLES       = ["dw.hitter_season_features", "dw.mlbplayers"]
PITCHER_TABLES      = ["dw.pitcher_pitch_type_features", "dw.mlbplayers",
                       "dw.pitch_type", "dw.pitch_core"]
ATBAT_TABLES        = ["dw.fact_atbat"]
ZONE_TABLES         = ["dw.pitcher_zone_counts", "dw.Zones", "dw.mlbplayers"]
PLAYER_STATS_TABLES = ["dw.player_stats"]
SEASON_TABLES       = ["dw.seasons"]
ROSTER_TABLES       = ["dw.rosters", "dw.teams"]
PITCH_CORE_TABLES   = ["dw.pitch_core", "dw.mlbplayers"]

FORBIDDEN = [
    "insert", "update", "delete", "drop",
    "alter", "truncate", "create", "exec", "--", ";--"
]

METADATA_FOLDER = "metadata"

roster_hallucination_keywords = [
    "player_events", "status_date", "roster_moves", "player_transactions",
    "mlbplayers",
    "to_char",
    "minor_league_team",
    "league__name",
]

ALWAYS_INCLUDE_TABLES = [
    "dw.teams",
    "dw.venues"
]

ALWAYS_KEEP_COLUMNS = {
    "dw.games": {
        "game_pk", "game_date", "season", "status__detailed_state",
        "series_description", "game_type", "day_night", "double_header",
        "teams__home__team__name", "teams__away__team__name",
        "teams__home__score", "teams__away__score",
        "teams__home__is_winner", "teams__away__is_winner",
        "rescheduled_from_date", "official_date",
    },
    "dw.rosters": {
        "person__full_name", "person__id", "season", "status__description",
        "status__code", "position__name", "parent_team_id", "jersey_number",
        "position__type", "position__abbreviation",
    },
    "dw.draft": {
        "person__full_name", "person__primary_position__name", "team__name",
        "year", "pick_number", "pick_round", "home__state", "home__city",
        "school__name", "signing_bonus",
    },
    "dw.teams": {
        "id", "name", "abbreviation", "league__name", "division__name",
    },
    "dw.mlbplayers": {
        "id", "full_name", "primary_position_name",
    },
    "dw.school_type_lookup": {
        "school_name", "school_type"
    },
    "dw.seasons": {
        "season_id", "has_wildcard", "regular_season_start_date", "all_star_date",
        "post_season_start_date", "qualifier_plate_appearances", "qualifier_outs_pitched"
    }
}

# ============================
# ORDERING RULES  (ORD-1)
# ============================

ORDERING_RULES = """
ORDERING RULES -- follow exactly, no exceptions:
- "best", "most", "highest", "largest", "top", "most wins"   → ORDER BY <metric> DESC
- "worst", "least", "lowest", "fewest", "bottom", "most losses" → ORDER BY <metric> ASC

CORRECT example -- worst home record (ORDER BY ASC):
  Q: which team does the San Francisco Giants have the worst home record against from 2020-2025
  SQL:
    SELECT TOP (100)
        teams__away__team__name                                             AS opponent,
        SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)        AS wins,
        SUM(CASE WHEN teams__away__is_winner = 1 THEN 1 ELSE 0 END)        AS losses,
        COUNT(*)                                                             AS games_played,
        CAST(
            SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)
            * 1.0 / NULLIF(COUNT(*), 0)
        AS decimal(5,3))                                                     AS win_pct
    FROM dw.games
    WHERE season BETWEEN '2020' AND '2025'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
      AND teams__home__team__name = 'San Francisco Giants'
    GROUP BY teams__away__team__name
    ORDER BY win_pct ASC;   -- ASC = worst record first

CORRECT example -- best home record (ORDER BY DESC):
  Q: which team does the San Francisco Giants have the best home record against from 2020-2025
  SQL:
    SELECT TOP (100)
        teams__away__team__name                                             AS opponent,
        SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)        AS wins,
        SUM(CASE WHEN teams__away__is_winner = 1 THEN 1 ELSE 0 END)        AS losses,
        COUNT(*)                                                             AS games_played,
        CAST(
            SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)
            * 1.0 / NULLIF(COUNT(*), 0)
        AS decimal(5,3))                                                     AS win_pct
    FROM dw.games
    WHERE season BETWEEN '2020' AND '2025'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
      AND teams__home__team__name = 'San Francisco Giants'
    GROUP BY teams__away__team__name
    ORDER BY win_pct DESC;  -- DESC = best record first
"""

# ============================
# DRAFT PROMPT TEMPLATE
# ============================

DRAFT_PROMPT_EXAMPLE = """
IMPORTANT: dw.draft is a FLAT table. It has NO related tables called dw.pick,
dw.team, dw.home, dw.player, or any other normalized sub-tables.
ALL columns you need are in dw.draft itself. Do NOT join to any invented table.
Do NOT join to dw.teams to get the team name — team__name is already on dw.draft.

To filter by school TYPE (high school, junior college, 4-year college):
  JOIN dw.school_type_lookup s ON s.school_name = d.school__name
  Then filter: s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')
  school_type values: 'HS' = High School, 'JC' = Junior/Community College, '4-Year' = College/University

CORRECT example — junior college picks (JOIN school_type_lookup):
  Q: junior college draft picks drafted in 2025
  SQL:
    SELECT TOP (100) d.person__full_name, d.person__primary_position__name,
                     d.school__name, s.school_type
    FROM dw.draft d
    JOIN dw.school_type_lookup s ON s.school_name = d.school__name
    WHERE d.year = 2025
      AND s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')
    ORDER BY d.pick_number;

CORRECT example — high school picks:
  Q: high school draft picks in 2025
  SQL:
    SELECT TOP (100) d.person__full_name, d.person__primary_position__name,
                     d.school__name, s.school_type
    FROM dw.draft d
    JOIN dw.school_type_lookup s ON s.school_name = d.school__name
    WHERE d.year = 2025
      AND s.school_type = 'HS'
    ORDER BY d.pick_number;

CORRECT example — name, position, single year:
  Q: name and position of 2025 draft picks from California
  SQL:
    SELECT TOP (100) d.person__full_name, d.person__primary_position__name
    FROM dw.draft d
    WHERE d.year = 2025
      AND d.home__state IN ('CA', 'California')
    ORDER BY d.pick_number;

CORRECT example — name, position, team, year range:
  Q: name, position and team of 2020-2025 draft picks from California
  SQL:
    SELECT TOP (100) d.person__full_name, d.person__primary_position__name,
                     d.team__name
    FROM dw.draft d
    WHERE d.year BETWEEN 2020 AND 2025
      AND d.home__state IN ('CA', 'California')
    ORDER BY d.year, d.pick_number;

CORRECT example — home state aggregation:
  Q: TOP 20 Home States with most draft picks
  SQL:
    SELECT TOP (20) d.home__state, COUNT(*) AS pick_count
    FROM dw.draft d
    WHERE d.home__state IS NOT NULL
    GROUP BY d.home__state
    ORDER BY pick_count DESC;

Column name reference for dw.draft (use ONLY these names):
  person__full_name              = player full name
  person__primary_position__name = player position
  home__state                    = home state abbreviation  e.g. 'CA', 'TX'
  home__city                     = home city
  team__name                     = drafting team name  <- already on dw.draft, NO JOIN needed
  school__name                   = school name (JOIN dw.school_type_lookup ON school_name to filter by school type)
  year                           = draft year integer  e.g. 2025
  pick_number                    = overall pick number
  pick_round                     = round number
  signing_bonus                  = signing bonus (nvarchar -- CAST to decimal)

Column name reference for dw.school_type_lookup:
  school_name  = matches d.school__name on dw.draft
  school_type  = 'HS' (high school) | 'JC' (junior/community college) | '4-Year' (college/university)

BANNED column names (do not use -- they do not exist):
  player_full_name, draft_date, pick_date, player_id, draft_id, team_id,
  home_id, home_state, team_name, position

BANNED joins (do not use -- dw.draft is self-contained except for school_type_lookup):
  JOIN dw.teams   -- NOT needed, use d.team__name directly
  JOIN dw.players -- does not exist
  JOIN dw.home    -- does not exist

STATE FILTER RULE:
  - ALWAYS use IN ('XX', 'Full State Name') e.g. IN ('CA', 'California')

SCHOOL TYPE FILTER RULE:
  - NEVER filter home__state for school type questions (junior college, high school, college)
  - ALWAYS JOIN dw.school_type_lookup and filter s.school_type for school type questions
  - "junior college", "juco", "community college" → s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')
  - "high school" → s.school_type = 'HS'
  - "college", "university", "4-year" → s.school_type = '4-Year'

SQL SERVER SYNTAX RULES (T-SQL only):
  - Use LIKE not ILIKE
  - Use TOP (N) not LIMIT N
  - Use YEAR(), MONTH(), DAY() not EXTRACT()
  - Never use NULLS FIRST or NULLS LAST
  - Never use :: cast operator
"""

# ============================
# HITTER PROMPT NOTES
# ============================

HITTER_PROMPT_NOTES = """
HITTER SEASON FEATURES RULES:
- Always JOIN dw.mlbplayers ON p.id = h.player_id to get player name
- Always filter plate_appearances >= 100 unless question asks for all players
- sprint_speed can be NULL -- always add IS NOT NULL when sorting by it
- wrc_plus: 100 = league average. Higher = better.
- To filter by team use h.team_name = 'San Francisco Giants' (column is on dw.hitter_season_features)

STANDARD ALIASES:
  h = dw.hitter_season_features
  p = dw.mlbplayers

CORRECT example:
  Q: top 20 hitters by exit velocity in 2025
  SQL:
    SELECT TOP (20) p.full_name, h.avg_exit_velocity, h.barrel_rate
    FROM dw.hitter_season_features h
    LEFT JOIN dw.mlbplayers p ON p.id = h.player_id
    WHERE h.season = 2025 AND h.plate_appearances >= 100
    ORDER BY h.avg_exit_velocity DESC;

CORRECT example -- filter by team:
  Q: hitter season stats for all San Francisco Giants in 2025
  SQL:
    SELECT TOP (100) p.full_name, h.season, h.plate_appearances,
                     h.avg_exit_velocity, h.barrel_rate, h.wrc_plus
    FROM dw.hitter_season_features h
    LEFT JOIN dw.mlbplayers p ON p.id = h.player_id
    WHERE h.season = 2025
      AND h.team_name = 'San Francisco Giants'
    ORDER BY h.plate_appearances DESC;
"""

# ============================
# PITCHER ZONE PROMPT NOTES
# ============================

ZONE_PROMPT_NOTES = """
PITCHER ZONE COUNTS RULES:
- Always JOIN dw.mlbplayers ON p.id = z.pitcher_id to get pitcher name
- Always JOIN dw.Zones ON zn.zone = z.zone to get zone description
- Always use SUM(z.pitch_count) when aggregating
- Zones 1-9 = in strike zone. Zones 11-14 = out of zone (chase zones)
"""

# ============================
# PLAYER STATS PROMPT NOTES
# ============================

PLAYER_STATS_PROMPT = """
PLAYER STATS RULES:
- stat__avg, stat__obp, stat__slg, stat__ops, stat__era, stat__whip stored as nvarchar.
  ALWAYS CAST before sorting: CAST(stat__avg AS decimal(10,3))
- Filter position__type = 'Hitter' for batting queries
- Filter position__type = 'Pitcher' for pitching queries
- Filter stat__plate_appearances >= 300 for batting title qualifiers
- Filter stat__games_started >= 10 for starting pitcher qualifiers
- Filter stat__games_started = 0 for reliever queries
- Use player__full_name for player name filters
- Use team__name for team name filters e.g. 'San Francisco Giants'

STANDARD ALIASES:
  ps = dw.player_stats
"""

# ============================
# PITCH CORE PROMPT NOTES  (TO-4)
# ============================

PITCH_CORE_PROMPT = """
PITCH CORE TABLE RULES -- dw.pitch_core:
- p.events contains the play outcome: 'single', 'double', 'triple', 'home_run',
  'field_out', 'strikeout', 'walk', 'hit_by_pitch', 'grounded_into_double_play', etc.
- "hits" means: events IN ('single', 'double', 'triple', 'home_run')
- p.pitcher_name  = pitcher full name (already flat on table, no JOIN to mlbplayers needed)
- p.stand         = batter handedness: 'L' or 'R'
- p.p_throws      = pitcher hand: 'L' or 'R'
- p.pitch_type    = pitch type code e.g. 'FF' (4-seam), 'SL' (slider), 'CH' (changeup),
                    'CU' (curveball), 'SI' (sinker), 'FC' (cutter), 'FS' (splitter)
- p.season        = integer season year e.g. 2025
- p.batter_name   = batter full name (flat on table)
- p.inning        = inning number
- p.balls / p.strikes = count at time of pitch

STANDARD ALIASES:
  p = dw.pitch_core

CORRECT example -- hits by pitch type and handedness:
  Q: pitcher, pitcher hand, batter side, pitch type for hits in 2025
  SQL:
    SELECT TOP (100)
        p.pitcher_name,
        p.p_throws          AS pitcher_hand,
        p.stand             AS batter_side,
        p.pitch_type,
        COUNT(*)            AS hit_count
    FROM dw.pitch_core p
    WHERE p.season = 2025
      AND p.events IN ('single', 'double', 'triple', 'home_run')
    GROUP BY p.pitcher_name, p.p_throws, p.stand, p.pitch_type
    ORDER BY hit_count DESC;

CORRECT example -- home runs by pitch type:
  Q: pitchers who allowed the most home runs by pitch type in 2025
  SQL:
    SELECT TOP (20) p.pitcher_name, p.pitch_type, COUNT(*) AS hr_allowed
    FROM dw.pitch_core p
    WHERE p.season = 2025 AND p.events = 'home_run'
    GROUP BY p.pitcher_name, p.pitch_type
    ORDER BY hr_allowed DESC;

BANNED columns (do not exist on dw.pitch_core):
  is_home_run, hit_flag, result_type
"""

# ============================
# GAMES PROMPT NOTES
# ============================

GAME_PROMPT_NOTES = """
GAMES TABLE COLUMN RULES -- dw.games:
Definitions:
- "record against" = wins and losses vs opponent
- "home record" = games where team is home team
- "best record" = highest win percentage (wins / (wins + losses))
- "worst record" = lowest win percentage (wins / (wins + losses))

BANNED columns (do not exist -- NEVER use them):
  original_game_date, original_date, scheduled_date, makeup_date,
  original_scheduled_date, postpone_date, rainout_date, is_home_run,
  teams__home__id,   (does not exist -- use teams__home__team__name directly)
  teams__away__id    (does not exist -- use teams__away__team__name directly)

FLAT COLUMNS on dw.games -- use these DIRECTLY, no JOIN to dw.teams needed:
  teams__home__team__name  = home team name (nvarchar)
  teams__away__team__name  = away team name (nvarchar)
  teams__home__score       = home score (int)
  teams__away__score       = away score (int)
  teams__home__is_winner   = 1 if home team won
  teams__away__is_winner   = 1 if away team won
  day_night                = 'day' or 'night'
  double_header            = 'Y' if doubleheader
  season                   = season year as nvarchar e.g. '2025'
  series_description       = 'Regular Season', 'Spring Training', etc.
  game_type                = 'R' regular, 'S' spring, 'D'/'L'/'W'/'F' postseason
  status__detailed_state   = 'Final', 'Postponed', 'Suspended', etc.
  rescheduled_from_date    = original date of a rescheduled game
  official_date            = new date after rescheduling

CRITICAL RULE -- FLAT COLUMNS ONLY for simple game queries:
  For any query that only needs team names, scores, counts, day_night,
  double_header, season, or series_description:
  -> DO NOT JOIN dw.teams. Use flat columns directly on dw.games.
  -> ONLY JOIN dw.teams when the query needs league__name or division__name.

WRONG pattern -- home record query (3 mistakes in one):
  SELECT g.game_pk, t_home.name, t_away.name,         <- game_pk in SELECT is wrong
         SUM(...) AS wins, SUM(...) AS losses,
         CAST(SUM(...) * 1.0 / NULLIF(COUNT(*), 0))   <- CAST missing AS type
  FROM dw.games g
  JOIN dw.teams t_home ON g.teams__home__team__id = t_home.id   <- column does not exist
  GROUP BY g.game_pk, t_home.name, t_away.name        <- GROUP BY game_pk is wrong

CORRECT pattern -- home record by team:
  SELECT g.teams__home__team__name AS home_team,
         COUNT(*) AS games,
         SUM(CASE WHEN g.teams__home__is_winner = 1 THEN 1 ELSE 0 END) AS wins,
         SUM(CASE WHEN g.teams__home__is_winner = 0 THEN 1 ELSE 0 END) AS losses,
         CAST(
             SUM(CASE WHEN g.teams__home__is_winner = 1 THEN 1 ELSE 0 END) * 1.0
             / NULLIF(COUNT(*), 0)
         AS decimal(5,3)) AS win_pct
  FROM dw.games g
  WHERE g.teams__home__team__name = 'San Francisco Giants'
    AND g.season BETWEEN '2020' AND '2025'
    AND g.series_description = 'Regular Season'
    AND g.status__detailed_state = 'Final'
  GROUP BY g.teams__home__team__name
  ORDER BY win_pct DESC;

  KEY RULES for home record pattern:
    - GROUP BY the team NAME column -- never GROUP BY game_pk
    - Never join dw.teams for team names -- use flat columns on dw.games directly
    - Never reference teams__home__team__id or teams__away__team__id -- they do not exist
    - CAST must always include AS decimal(5,3) -- bare CAST() is a syntax error

WRONG pattern (do NOT do this for simple queries):
  SELECT COUNT(*) FROM dw.games g
  JOIN dw.teams t_home ON g.teams__home__team__id = t_home.id    <- unnecessary JOIN
  JOIN dw.teams t_away ON g.teams__away__team__id = t_away.id    <- unnecessary JOIN
  WHERE YEAR(g.game_date) = 2025 ...

HOME RUNS in dw.pitch_core: use events = 'home_run' -- NOT is_home_run (does not exist).

CORRECT example -- game count (no JOIN, no TOP, use season column):
  Q: how many games were played in the 2025 regular season
  SQL:
    SELECT COUNT(*) AS game_count
    FROM dw.games
    WHERE season = '2025'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final';

CORRECT example -- Opening Day games (derive from MIN game_date):
  Q: game_pk, home team name, away team name for all games on Opening Day in 2024
  SQL:
    SELECT TOP (100)
        game_pk,
        teams__home__team__name AS home_team,
        teams__away__team__name AS away_team,
        game_date
    FROM dw.games
    WHERE season = '2024'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
      AND CAST(game_date AS date) = (
          SELECT MIN(CAST(game_date AS date))
          FROM dw.games
          WHERE season = '2024'
            AND series_description = 'Regular Season'
            AND status__detailed_state = 'Final'
      )
    ORDER BY game_date;

CORRECT example -- date query with flat columns (no JOIN):
  Q: game_pk, home team, away team, home score, away score for all games on 2025-04-15
  SQL:
    SELECT TOP (100) game_pk,
                     teams__home__team__name AS home_team,
                     teams__away__team__name AS away_team,
                     teams__home__score      AS home_score,
                     teams__away__score      AS away_score
    FROM dw.games
    WHERE CAST(game_date AS date) = '2025-04-15'
      AND status__detailed_state = 'Final'
    ORDER BY game_date;

CORRECT example -- day/night breakdown (GROUP BY flat column, no JOIN):
  Q: all night games in 2025 regular season count
  SQL:
    SELECT day_night, COUNT(*) AS game_count
    FROM dw.games
    WHERE season = '2025'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
    GROUP BY day_night
    ORDER BY game_count DESC;

CORRECT example -- doubleheader count by team (flat column, no JOIN):
  Q: count of double header games in 2025 by team
  SQL:
    SELECT teams__home__team__name AS team, COUNT(*) AS doubleheader_games
    FROM dw.games
    WHERE season = '2025'
      AND double_header = 'Y'
      AND status__detailed_state = 'Final'
    GROUP BY teams__home__team__name
    ORDER BY doubleheader_games DESC;

CORRECT example -- postseason series (ORDER BY MIN(game_date) not game_date):
  Q: number of games in each postseason series in 2025
  SQL:
    SELECT series_description,
           teams__home__team__name AS home_team,
           teams__away__team__name AS away_team,
           COUNT(*) AS games_played
    FROM dw.games
    WHERE season = '2025'
      AND game_type IN ('D','L','W','F')
      AND status__detailed_state = 'Final'
    GROUP BY series_description, teams__home__team__name, teams__away__team__name
    ORDER BY MIN(game_date);

CORRECT example -- season range with BETWEEN (use g.season, nvarchar, quoted):
  Q: how many regular season games were played from 2022 to 2024
  SQL:
    SELECT g.season, COUNT(*) AS game_count
    FROM dw.games g
    WHERE g.season BETWEEN '2022' AND '2024'
      AND g.series_description = 'Regular Season'
      AND g.status__detailed_state = 'Final'
    GROUP BY g.season
    ORDER BY g.season;

CORRECT example -- worst record against opponent (ORDER BY ASC):
  Q: which team does the San Francisco Giants have the worst home record against from 2020-2025
  SQL:
    SELECT TOP (100)
        teams__away__team__name                                             AS opponent,
        SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)        AS wins,
        SUM(CASE WHEN teams__away__is_winner = 1 THEN 1 ELSE 0 END)        AS losses,
        COUNT(*)                                                             AS games_played,
        CAST(
            SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)
            * 1.0 / NULLIF(COUNT(*), 0)
        AS decimal(5,3))                                                     AS win_pct
    FROM dw.games
    WHERE season BETWEEN '2020' AND '2025'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
      AND teams__home__team__name = 'San Francisco Giants'
    GROUP BY teams__away__team__name
    ORDER BY win_pct ASC;   -- ASC = worst record first

CORRECT example -- win/loss record against opponents (ORDER BY DESC):
  Q: record, opponent the San Francisco Giants have the best record against since 1960
  SQL:
    SELECT TOP (100)
        teams__home__team__name                                           AS team,
        teams__away__team__name                                           AS opponent,
        SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)      AS wins,
        SUM(CASE WHEN teams__away__is_winner = 1 THEN 1 ELSE 0 END)      AS losses,
        COUNT(*)                                                           AS games_played,
        CAST(
            SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)
            * 1.0 / NULLIF(COUNT(*), 0)
        AS decimal(5,3))                                                   AS win_pct
    FROM dw.games
    WHERE season >= '1960'
      AND series_description = 'Regular Season'
      AND status__detailed_state = 'Final'
      AND teams__home__team__name = 'San Francisco Giants'
    GROUP BY teams__home__team__name, teams__away__team__name
    ORDER BY win_pct DESC;  -- DESC = best record first

  KEY RULES for record-against pattern:
    - NEVER join dw.teams to get team names -- use flat columns directly
    - NEVER join dw.teams to get division__name for a record query -- not needed
    - NEVER use teams__home__id or teams__away__id -- they do not exist
    - ALWAYS use GROUP BY opponent + SUM(CASE WHEN is_winner) to compute record
    - ALWAYS use NULLIF(COUNT(*), 0) in win_pct to avoid divide-by-zero
    - Use season >= 'YYYY' not YEAR(game_date) >= YYYY
"""

# ============================
# ROSTER PROMPT NOTES
# ============================

ROSTER_PROMPT_COMPACT = """
ROSTER QUERY -- USE ONLY THESE TWO TABLES:
  FROM dw.rosters r
  JOIN dw.teams t ON t.id = r.parent_team_id

KEY COLUMNS:
  r.person__full_name    -- player name
  r.position__name       -- position
  r.status__description  -- use this for filtering (see values below)
  r.season               -- INTEGER, never quote: r.season = 2025
  t.name                 -- team name e.g. 'San Francisco Giants'

CRITICAL: ALL column references in WHERE and ORDER BY MUST use the r. or t. alias.
NEVER write bare: season, status__description, person__full_name
ALWAYS write:     r.season, r.status__description, r.person__full_name

STATUS VALUES: 'Active', 'Released', 'Waived', 'Traded',
  'Designated for Assignment', 'Injured 10-Day', 'Injured 15-Day','Injured 60-Day',
  'Free Agent', 'Optioned to Minors', 'Reassigned to Minors',
  'Bereavement List', 'Paternity List', 'Suspended List',
  'Restricted List', 'Voluntarily Retired'

BANNED: dw.games, dw.mlbplayers, status_date, player_id

EXAMPLE 1 -- simple filter:
  Q: players reassigned to minors in 2025
  SQL:
    SELECT TOP (100) r.person__full_name, t.name AS team_name,
                     r.position__name, r.status__description, r.season
    FROM dw.rosters r
    JOIN dw.teams t ON t.id = r.parent_team_id
    WHERE r.season = 2025
      AND r.status__description = 'Reassigned to Minors'
    ORDER BY t.name, r.person__full_name;

EXAMPLE 2 -- roster additions (players on new season but NOT prior season):
  Q: roster changes between 2025 and 2026 for the New York Mets players on 2026 but not 2025
  SQL:
    SELECT r26.person__full_name, r26.position__name, r26.jersey_number
    FROM dw.rosters r26
    JOIN dw.teams t ON t.id = r26.parent_team_id
    WHERE t.name = 'New York Mets'
      AND r26.season = 2026
      AND r26.status__code = 'A'
      AND r26.person__id NOT IN (
          SELECT r25.person__id
          FROM dw.rosters r25
          JOIN dw.teams t2 ON t2.id = r25.parent_team_id
          WHERE t2.name = 'New York Mets'
            AND r25.season = 2025
      )
    ORDER BY r26.position__type, r26.person__full_name;

EXAMPLE 3 -- injured list filter (use status__description, NOT status__code):
  Q: players on the 10-day, 15-day or 60-day injured list for the Giants in 2024-2026
  SQL:
    SELECT TOP (100) r.person__full_name, t.name AS team_name,
                     r.position__name, r.status__description, r.season
    FROM dw.rosters r
    JOIN dw.teams t ON t.id = r.parent_team_id
    WHERE t.name = 'San Francisco Giants'
      AND r.season BETWEEN 2024 AND 2026
      AND r.status__description IN ('Injured 10-Day', 'Injured 15-Day', 'Injured 60-Day')
    ORDER BY r.season DESC, r.status__description, r.person__full_name;
"""

# ============================
# OPT-3: Prompt block dispatch
# ============================

PROMPT_BLOCKS = {
    "roster":       ROSTER_PROMPT_COMPACT,
    "draft":        DRAFT_PROMPT_EXAMPLE,
    "hitter":       HITTER_PROMPT_NOTES,
    "zone":         ZONE_PROMPT_NOTES,
    "player_stats": PLAYER_STATS_PROMPT,
    "pitch_core":   PITCH_CORE_PROMPT,   # TO-4
    "game":         GAME_PROMPT_NOTES,
}


def get_query_type(question: str) -> str:
    if re.search(
        r'\broster\b|\breleased\b|\bwaived\b|\btraded\b|\bdesignated\b'
        r'|\binjured.list\b|\b10.day\b|\b15.day\b|\b60.day\b'
        r'|\bfree.agent\b|\breassigned\b|\bminors\b|\bclaimed\b'
        r'|\bbereavement\b|\bpaternity\b|\bsuspended\b|\brestricted.list\b'
        r'|\bnon.roster\b|\bvoluntarily.retired\b|\badministrative.leave\b'
        r'|\btemporary.inactive\b|\bineligible.list\b|\breserve.list\b',
        question, re.IGNORECASE
    ):
        return "roster"

    if re.search(r'\bdraft', question, re.IGNORECASE):
        return "draft"

    # TO-5: added hitter.season and season.stats
    if re.search(
        r'\bhitter\b|\bexit.velocit|\bbarrel\b|\bwrc\b|\blaunch.angle\b'
        r'|\bhard.hit\b|\bsprint.speed\b|\biso\b|\bhitter_season\b'
        r'|\bhitter.season\b|\bseason.stats\b',
        question, re.IGNORECASE
    ):
        return "hitter"

    if re.search(
        r'\bzone\b|\bpitch.location\b|\bheart.of.the.plate\b|\bchase\b'
        r'|\bpitcher_zone\b|\bin.zone\b|\bout.of.zone\b',
        question, re.IGNORECASE
    ):
        return "zone"

    if re.search(
        r'\bhome.run\b|\bbatting.average\b|\bera\b|\bwhip\b|\brbi\b'
        r'|\bstolen.base\b|\bops\b|\bstarter\b|\breliever\b'
        r'|\bplayer.stats\b|\bpitching.stat\b|\bhitting.stat\b'
        r'|\bstat__\b|\bleader\b|\bsaves\b|\bshutout\b|\bcomplete.game\b'
        r'|\bwild.pitch\b|\bfielding.percentage\b|\bstrikeout.leader\b'
        r'|\bmost.wins\b|\bwin.leader\b|\bpitcher.*wins\b|\bwins.*pitcher\b'
        r'|\bmost wins\b|\bpitchers.*wins\b',
        question, re.IGNORECASE
    ):
        return "player_stats"

    # TO-4: pitch_core routing before game fallback
    if re.search(
        r'\bpitch.type\b|\bpitcher.hand\b|\bbatter.side\b|\bp_throws\b'
        r'|\bpitch_core\b|\bhits\b.*\bpitch\b|\bpitch\b.*\bhits\b'
        r'|\bevents\b.*\bpitch\b|\bpitch\b.*\bevents\b',
        question, re.IGNORECASE
    ):
        return "pitch_core"

    return "game"


# ============================
# LOGGING SETUP
# ============================

logging.basicConfig(
    filename="mlb_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

# ============================
# REGEX
# ============================

SELECT_REGEX = re.compile(r"\bselect\b", re.IGNORECASE)
TOP_REGEX    = re.compile(r"\bselect\s+(distinct\s+)?top\s*\(\d+\)", re.IGNORECASE)

TABLE_ALIAS_REGEX = re.compile(
    r"(?:from|join)\s+((?:dw\.)?\w+)\s+(?:as\s+)?(\w+)",
    re.IGNORECASE
)

COLUMN_REGEX = re.compile(r"(\w+)\.(\w+)")

POSTGRES_PATTERNS = [
    (re.compile(r'\bto_char\s*\(',    re.IGNORECASE), "to_char() is PostgreSQL -- use FORMAT() or CONVERT() for SQL Server"),
    (re.compile(r'\bto_date\s*\(',    re.IGNORECASE), "to_date() is PostgreSQL -- use CONVERT(date, ...) for SQL Server"),
    (re.compile(r'\bto_number\s*\(',  re.IGNORECASE), "to_number() is PostgreSQL -- use CAST() or TRY_CAST() for SQL Server"),
    (re.compile(r'\bdate_trunc\s*\(', re.IGNORECASE), "date_trunc() is PostgreSQL -- use DATEADD/DATEDIFF for SQL Server"),
    (re.compile(r'\bextract\s*\(',    re.IGNORECASE), "EXTRACT() is PostgreSQL -- use YEAR(), MONTH(), DAY() for SQL Server"),
    (re.compile(r'\bnulls\s+(?:first|last)\b', re.IGNORECASE), "NULLS FIRST/LAST is PostgreSQL -- use CASE in ORDER BY for SQL Server"),
    (re.compile(r'::\w+',             re.IGNORECASE), ":: cast operator is PostgreSQL -- use CAST() or CONVERT() for SQL Server"),
]

SQL_RESERVED_WORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "on", "and", "or", "not", "in", "is", "null", "as", "by", "group",
    "order", "having", "top", "distinct", "case", "when", "then", "else",
    "end", "like", "between", "exists", "union", "all", "asc", "desc",
    "set", "with", "over", "partition", "row_number", "count", "sum",
    "avg", "min", "max", "year", "month", "day", "cast", "convert",
    "coalesce", "isnull", "nullif", "len", "upper", "lower", "trim",
    "substring", "charindex", "replace", "getdate", "dateadd", "datediff",
    "dw"
}

# ============================
# MLB TEAM NAMES
# ============================

MLB_TEAMS = [
    "arizona diamondbacks", "atlanta braves", "baltimore orioles",
    "boston red sox", "chicago cubs", "chicago white sox",
    "cincinnati reds", "cleveland guardians", "colorado rockies",
    "detroit tigers", "houston astros", "kansas city royals",
    "los angeles angels", "los angeles dodgers", "miami marlins",
    "milwaukee brewers", "minnesota twins", "new york mets",
    "new york yankees", "oakland athletics", "philadelphia phillies",
    "pittsburgh pirates", "san diego padres", "san francisco giants",
    "seattle mariners", "st. louis cardinals", "tampa bay rays",
    "texas rangers", "toronto blue jays", "washington nationals",
]

# ============================
# REQUEST MODELS
# ============================

class QuestionRequest(BaseModel):
    question: str

class ReflectQueryRequest(BaseModel):
    question: str

# ============================
# ENGINE + LLM
# ============================

def create_agent():
    conn_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=KEITH-PERSONAL;"
        "DATABASE=dlt;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    params = urllib.parse.quote_plus(conn_string)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True,
        pool_recycle=300
    )
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        request_timeout=120,
        num_ctx=2048,
        num_predict=8000
    )
    logger.info(f"Ollama LLM created: model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}")
    return engine, llm


# ============================
# HELPERS
# ============================

def trim_to_token_budget(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    result = []
    total = 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        result.append(line)
        total += len(line) + 1
    return "\n".join(result) + "\n... [truncated]"


def sql_contains_year(sql: str, year: str) -> bool:
    return bool(re.search(r'(?<!\w)' + re.escape(year) + r'(?!\w)', sql))


def sql_matches_question_context(matched_sql: str, question: str, y1: str, y2: str) -> bool:
    sql_lower = matched_sql.lower()

    has_between = (
        f"between '{y1}'" in sql_lower and y2 in sql_lower or
        f"between {y1}"   in sql_lower and y2 in sql_lower or
        f">= '{y1}'"      in sql_lower or
        f">= {y1}"        in sql_lower
    )
    years_ok = (
        (sql_contains_year(matched_sql, y1) and sql_contains_year(matched_sql, y2))
        or has_between
    )
    if not years_ok:
        return False

    question_lower = question.lower()
    for team in MLB_TEAMS:
        if team in question_lower:
            if team not in sql_lower:
                return False
            break

    return True


def fix_ambiguous_columns(sql: str) -> str:
    if not (re.search(r'\bdw\.rosters\b', sql, re.IGNORECASE) and
            re.search(r'\bdw\.teams\b',   sql, re.IGNORECASE)):
        return sql

    ROSTER_COLS = [
        "season", "status__description", "status__code",
        "person__full_name", "position__name", "position__code",
        "position__type", "position__abbreviation", "jersey_number",
        "parent_team_id", "person__id",
    ]

    for col in ROSTER_COLS:
        sql = re.sub(
            r'(?<![.\w])(' + re.escape(col) + r')(?!\w)',
            r'r.\1',
            sql,
            flags=re.IGNORECASE
        )

    return sql


def enrich_question(question: str) -> str:
    q = question

    is_draft = bool(re.search(r'\bdraft', q, re.IGNORECASE))

    is_player_stats = bool(re.search(
        r'\bhome.run\b|\bbatting.average\b|\bera\b|\bwhip\b|\brbi\b'
        r'|\bstolen.base\b|\bops\b|\bstarter\b|\breliever\b'
        r'|\bplayer.stats\b|\bpitching.stat\b|\bhitting.stat\b'
        r'|\bstat__\b|\bleader\b|\bsaves\b|\bshutout\b|\bcomplete.game\b'
        r'|\bwild.pitch\b|\bfielding.percentage\b|\bstrikeout.leader\b'
        r'|\bmost.wins\b|\bwin.leader\b|\bpitcher.*wins\b|\bwins.*pitcher\b',
        question, re.IGNORECASE))

    # TO-5: added hitter.season and season.stats
    is_hitter = bool(re.search(
        r'\bhitter\b|\bexit.velocit|\bbarrel\b|\bwrc\b|\bseason_features\b'
        r'|\bhitter.season\b|\bseason.stats\b',
        q, re.IGNORECASE))

    is_pitch = bool(re.search(
        r'\bpitch\b|\bpitcher\b|\bstrikeout\b|\bspin\b|\bwhiff\b',
        q, re.IGNORECASE)) and not is_player_stats

    is_roster = bool(re.search(
        r'\broster\b|\breleased\b|\bwaived\b|\btraded\b|\bdesignated\b'
        r'|\binjured.list\b|\b10.day\b|\b15.day\b|\b60.day\b'
        r'|\bfree.agent\b|\breassigned\b|\bminors\b|\bclaimed\b'
        r'|\bbereavement\b|\bpaternity\b|\bsuspended\b|\brestricted.list\b'
        r'|\bnon.roster\b|\bvoluntarily.retired\b|\badministrative.leave\b'
        r'|\btemporary.inactive\b|\bineligible.list\b|\breserve.list\b',
        q, re.IGNORECASE))

    is_jc = bool(re.search(
        r'\bjunior.college\b|\bjuco\b|\bcommunity.college\b|\bjc\s+draft\b|\bjc\s+pick\b',
        q, re.IGNORECASE))

    range_match = re.search(r'\b(\d{4})\s*(?:-|to)\s*(\d{4})\b', q, re.IGNORECASE)

    if is_draft:
        if is_jc:
            q += (
                " [JOIN dw.school_type_lookup s ON s.school_name = d.school__name."
                " Filter: s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')."
                " Do NOT filter home__state for school type questions.]"
            )
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += (f" [Use WHERE d.year BETWEEN {y1} AND {y2} on dw.draft."
                  f" Do NOT use draft_date or pick_date.]")
        else:
            year_match = re.search(
                r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b|\b(\d{4})\s+draft\b',
                q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2) or year_match.group(3)
                q += (f" [Use WHERE d.year = {year} on dw.draft."
                      f" Do NOT use draft_date or pick_date.]")

    elif is_hitter:
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += f" [Use WHERE h.season BETWEEN {y1} AND {y2} on dw.hitter_season_features.]"
        else:
            year_match = re.search(r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b', q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                q += f" [Use WHERE h.season = {year} on dw.hitter_season_features.]"

        # TO-6: team name hint for hitter queries
        team_match = re.search(
            r'(arizona diamondbacks|atlanta braves|baltimore orioles|boston red sox'
            r'|chicago cubs|chicago white sox|cincinnati reds|cleveland guardians'
            r'|colorado rockies|detroit tigers|houston astros|kansas city royals'
            r'|los angeles angels|los angeles dodgers|miami marlins|milwaukee brewers'
            r'|minnesota twins|new york mets|new york yankees|oakland athletics'
            r'|philadelphia phillies|pittsburgh pirates|san diego padres'
            r'|san francisco giants|seattle mariners|st\. louis cardinals'
            r'|tampa bay rays|texas rangers|toronto blue jays|washington nationals)',
            q, re.IGNORECASE)
        if team_match:
            team = team_match.group(1).title()
            q += (f" [Filter by team: use h.team_name = '{team}' on dw.hitter_season_features."
                  f" Do NOT join dw.rosters for team filtering.]")

    elif is_pitch:
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += f" [Use WHERE p.season BETWEEN {y1} AND {y2} on dw.pitch_core or dw.pitcher_pitch_type_features.]"
        else:
            year_match = re.search(r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b', q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                q += f" [Use WHERE p.season = {year}.]"

    elif is_player_stats:
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += f" [Use WHERE ps.season BETWEEN '{y1}' AND '{y2}' on dw.player_stats.]"
        else:
            year_match = re.search(r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b', q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                q += f" [Use WHERE ps.season = '{year}'.]"

    elif is_roster:
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += (f" [Use WHERE r.season BETWEEN {y1} AND {y2} on dw.rosters."
                  f" JOIN dw.teams t ON t.id = r.parent_team_id."
                  f" Do NOT join dw.games. Do NOT use status_date or event_type_code.]")
        else:
            year_match = re.search(r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b', q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                q += (f" [Use WHERE r.season = {year} on dw.rosters."
                      f" JOIN dw.teams t ON t.id = r.parent_team_id."
                      f" Do NOT join dw.games. Do NOT use status_date.]")

    else:
        if range_match:
            y1, y2 = range_match.group(1), range_match.group(2)
            q += (f" [Use WHERE g.season BETWEEN '{y1}' AND '{y2}' on dw.games."
                  f" Do NOT use YEAR(game_date). Do NOT use unqualified 'season'.]")
        else:
            year_match = re.search(r'\bin\s+(\d{4})\b|\bduring\s+(\d{4})\b', q, re.IGNORECASE)
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                q += f" [Use WHERE g.season = '{year}' on dw.games.]"

        # TO-2: Opening Day enrich hint
        if re.search(r'\bopening.day\b', q, re.IGNORECASE):
            yr_match = re.search(r'\b(20\d{2})\b', q)
            yr = yr_match.group(1) if yr_match else None
            if yr:
                q += (
                    f" [Opening Day = first regular season game date in {yr}."
                    f" Use: AND CAST(game_date AS date) = (SELECT MIN(CAST(game_date AS date))"
                    f" FROM dw.games WHERE season = '{yr}'"
                    f" AND series_description = 'Regular Season'"
                    f" AND status__detailed_state = 'Final')]"
                )

        has_count  = bool(re.search(r'\bhow many\b', q, re.IGNORECASE))
        has_detail = bool(re.search(r'\bgame_pk\b|\bwinner\b|\bgame date\b', q, re.IGNORECASE))
        if has_count and has_detail:
            q += " [Return one row per game with detail columns. Do NOT use COUNT(*) or GROUP BY.]"

        team_match = re.search(
            r'(san francisco giants|los angeles dodgers|new york yankees|boston red sox)',
            q, re.IGNORECASE)
        if team_match:
            q += " [Filter by team names using two separate JOINs: t_home and t_away with OR condition for home/away.]"

    return q


def validate_groupby(sql: str):
    sql_lower = sql.lower()
    if "group by" not in sql_lower:
        return
    has_aggregate = any(fn in sql_lower for fn in ("count(", "sum(", "min(", "max(", "avg("))
    if not has_aggregate:
        raise ValueError(
            "GROUP BY used without aggregate function. "
            "Remove GROUP BY or add COUNT(*)/SUM()."
        )
    if re.search(r'\border\s+by\b.*\bgame_date\b', sql_lower):
        gb_match = re.findall(r'group\s+by\s+(.*?)(?:order|having|$)', sql_lower, re.DOTALL)
        if gb_match and 'game_date' not in gb_match[0]:
            if not re.search(r'\b(?:min|max)\s*\(\s*game_date\s*\)', sql_lower):
                raise ValueError(
                    "ORDER BY game_date is invalid inside a GROUP BY query. "
                    "Use ORDER BY MIN(game_date) instead."
                )
    score_cols = ["teams__home__score", "teams__away__score"]
    for col in score_cols:
        if col in sql_lower and "case" not in sql_lower:
            raise ValueError(
                f"Column {col} used with GROUP BY but not in aggregate. "
                "Wrap in CASE or remove GROUP BY."
            )


def validate_tsql(sql: str):
    for pattern, message in POSTGRES_PATTERNS:
        if pattern.search(sql):
            raise ValueError(
                f"Invalid SQL Server syntax: {message}. "
                "Rewrite using T-SQL / Microsoft SQL Server syntax only."
            )


def validate_draft_no_join(sql: str):
    if re.search(r'\bdw\.draft\b', sql, re.IGNORECASE):
        if re.search(r'\bjoin\s+dw\.teams\b', sql, re.IGNORECASE):
            raise ValueError(
                "Draft query must NOT join dw.teams. "
                "Use d.team__name directly from dw.draft -- no JOIN needed."
            )


def validate_no_game_pk_in_group_by(sql: str):
    if re.search(r'\bgroup\s+by\b.*\bgame_pk\b', sql, re.IGNORECASE | re.DOTALL):
        raise ValueError(
            "GROUP BY game_pk is wrong for aggregate queries. "
            "GROUP BY should be on the team name column, not game_pk."
        )


def validate_cast_has_type(sql: str):
    for match in re.finditer(r'\bCAST\s*\(', sql, re.IGNORECASE):
        depth = 1
        i = match.end()
        while i < len(sql) and depth > 0:
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
            i += 1
        cast_body = sql[match.start():i]
        if not re.search(r'\bAS\s+\w', cast_body, re.IGNORECASE):
            raise ValueError(
                f"CAST missing AS type near: {cast_body[:80]}. "
                "Use CAST(expr AS decimal(5,3)) or CAST(expr AS nvarchar)."
            )


def validate_no_fake_team_id_join(sql: str):
    if re.search(r'teams__(?:home|away)__team__id', sql, re.IGNORECASE):
        raise ValueError(
            "FAKE_TEAM_ID_JOIN: teams__home__team__id / teams__away__team__id does not exist. "
            "Use flat columns teams__home__team__name / teams__away__team__name directly "
            "on dw.games -- no JOIN to dw.teams needed for team name lookups."
        )


def validate_ordering_direction(sql: str, question: str):
    worst_keywords = [
        "worst", "lowest", "fewest", "least", "bottom", "most losses",
        "worst record", "fewest wins"
    ]
    best_keywords = [
        "best", "highest", "most wins", "top", "largest", "most home runs",
        "best record", "most strikeouts", "most saves"
    ]

    question_lower = question.lower()
    is_worst = any(kw in question_lower for kw in worst_keywords)
    is_best  = any(kw in question_lower for kw in best_keywords)

    if is_worst and not is_best:
        if re.search(r'\bORDER\s+BY\b.*\bDESC\b', sql, re.IGNORECASE | re.DOTALL):
            raise ValueError(
                "Question asks for 'worst/lowest/fewest' but SQL uses ORDER BY DESC. "
                "Change to ORDER BY ASC -- worst = lowest value = ASC."
            )

    if is_best and not is_worst:
        if re.search(r'\bORDER\s+BY\b.*\bASC\b', sql, re.IGNORECASE | re.DOTALL):
            raise ValueError(
                "Question asks for 'best/highest/most' but SQL uses ORDER BY ASC. "
                "Change to ORDER BY DESC -- best = highest value = DESC."
            )


def validate_school_type_filter(sql: str, question: str):
    is_jc_question = bool(re.search(
        r'\bjunior.college\b|\bjuco\b|\bcommunity.college\b|\bjc\s+draft\b|\bjc\s+pick\b',
        question, re.IGNORECASE
    ))
    if not is_jc_question:
        return

    has_school_type_filter = bool(re.search(
        r'school_type_lookup|school_type|school__name\s+LIKE',
        sql, re.IGNORECASE
    ))
    if not has_school_type_filter:
        raise ValueError(
            "JC_FILTER_MISSING: Question asks for junior college picks but SQL has no school type filter. "
            "JOIN dw.school_type_lookup s ON s.school_name = d.school__name "
            "and filter s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')."
        )


# ============================
# OPT-4: Column relevance filtering
# ============================

def filter_columns_for_question(question: str, table: str, cols: dict, cap: int) -> dict:
    always = ALWAYS_KEEP_COLUMNS.get(table, set())
    keywords = set(re.findall(r'\b\w+\b', question.lower()))

    filtered = {}
    for col, dtype in cols.items():
        col_words = set(re.split(r'[_\W]+', col.lower()))
        if col in always or col_words & keywords:
            filtered[col] = dtype
        if len(filtered) >= cap:
            break

    for col in always:
        if col in cols and col not in filtered:
            filtered[col] = cols[col]

    return filtered


# ============================
# OPT-2: Semantic example cache
# ============================

def _rebuild_embeddings(app_state):
    if not SEMANTIC_AVAILABLE:
        return
    examples = app_state.metadata.get("examples", {}).get("examples", [])
    if not examples:
        return
    questions = [ex["question"] for ex in examples]
    app_state.example_embeddings = app_state.embedder.encode(
        questions, normalize_embeddings=True
    )
    app_state.example_list = examples
    logger.info(f"Semantic index rebuilt: {len(examples)} examples")


def find_best_example(question: str, metadata: dict, threshold: float = 0.99):
    if SEMANTIC_AVAILABLE and hasattr(app, "state") and hasattr(app.state, "embedder"):
        example_list = getattr(app.state, "example_list", [])
        embeddings   = getattr(app.state, "example_embeddings", None)

        if example_list and embeddings is not None:
            import numpy as np
            q_emb = app.state.embedder.encode([question], normalize_embeddings=True)
            scores = np.dot(embeddings, q_emb.T).flatten()
            best_idx   = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= threshold:
                matched = example_list[best_idx]
                logger.info(
                    f"Semantic match ({best_score:.0%}): '{matched['question']}' -- skipping LLM"
                )
                return matched["sql"], best_score

            logger.debug(f"No semantic match (best: {best_score:.0%}) -- calling LLM")
            return None, best_score

    # Fallback: SequenceMatcher
    if "examples" not in metadata:
        return None, 0.0

    question_clean = question.strip().lower()
    best_score     = 0
    best_sql       = None
    best_question  = None

    for ex in metadata["examples"].get("examples", []):
        ex_question = ex.get("question", "").strip().lower()
        score = SequenceMatcher(None, question_clean, ex_question).ratio()
        if score > best_score:
            best_score    = score
            best_sql      = ex["sql"]
            best_question = ex_question

    sm_threshold = 0.99
    if best_score >= sm_threshold:
        logger.info(f"SequenceMatcher match ({best_score:.0%}): '{best_question}' -- skipping LLM")
        return best_sql, best_score

    logger.debug(f"No SequenceMatcher match (best score: {best_score:.0%}) -- calling LLM")
    return None, best_score


# ============================
# OPT-6: Auto-promote LLM results to examples
# TO-7: Fixed dead code bug -- JC guard was unreachable
# ============================

def maybe_save_example(question: str, sql: str, source: str, rows: list):
    if source != "llm":
        return
    if not rows or len(rows) >= AUTO_TOP_LIMIT:
        return

    # Don't save if the SQL looks wrong for the question  (TO-7: fixed indentation)
    is_jc_question = bool(re.search(
        r'\bjunior.college\b|\bjuco\b|\bcommunity.college\b', question, re.IGNORECASE
    ))
    has_jc_filter = bool(re.search(
        r'school_type_lookup|school_type|school__name\s+LIKE', sql, re.IGNORECASE
    ))
    if is_jc_question and not has_jc_filter:
        logger.warning("maybe_save_example: skipping bad JC example -- no school_type filter in SQL")
        return

    path = os.path.join(METADATA_FOLDER, "examples.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"examples": []}
    except json.JSONDecodeError as e:
        logger.error(f"maybe_save_example: JSON parse error in {path}: {e}")
        return

    existing = {ex.get("question", "").strip().lower() for ex in data["examples"]}
    if question.strip().lower() in existing:
        return

    data["examples"].append({"question": question, "sql": sql})

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Auto-saved new example: '{question}'")

        app.state.metadata["examples"] = data
        _rebuild_embeddings(app.state)

    except Exception as e:
        logger.error(f"maybe_save_example: failed to write {path}: {e}")


async def check_ollama():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}", timeout=5)
            logger.info(f"Ollama reachable: {r.status_code}")
    except Exception as e:
        logger.error(f"Ollama is NOT reachable at {OLLAMA_BASE_URL}: {e}")
        raise RuntimeError(f"Ollama not running at {OLLAMA_BASE_URL}. Start with: ollama serve")


async def warmup_model():
    try:
        print(f"    Warming up '{OLLAMA_MODEL}'...")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": "hi",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "num_ctx":     128,
                        "temperature": 0,
                    }
                },
                timeout=60
            )
        print(f"    Model '{OLLAMA_MODEL}' is warm and ready.")
    except Exception as e:
        logger.warning(f"Model warm-up failed (non-fatal): {e}")


# ============================
# REFLECTION HELPERS
# ============================

def reflect_classify_question(question: str, metadata: dict, allowed_tables: dict) -> dict:
    query_type    = get_query_type(question)
    enriched      = enrich_question(question)
    filtered_tables = filter_relevant_tables(enriched, allowed_tables)
    matched_sql, match_score = find_best_example(question, metadata, threshold=0.99)

    all_scores = []
    if "examples" in metadata:
        question_clean = question.strip().lower()
        for ex in metadata["examples"].get("examples", []):
            score = SequenceMatcher(
                None,
                question_clean,
                ex.get("question", "").strip().lower()
            ).ratio()
            all_scores.append({
                "example_question": ex.get("question", ""),
                "similarity_score": round(score, 4),
                "would_match": score >= 0.99
            })
        all_scores.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {
        "question":              question,
        "enriched_question":     enriched,
        "query_type":            query_type,
        "path":                  "example_cache" if matched_sql else "llm",
        "example_match": {
            "found":     matched_sql is not None,
            "score":     round(match_score, 4),
            "threshold": 0.99,
            "sql":       matched_sql
        },
        "top_example_scores":    all_scores[:5],
        "tables_selected":       list(filtered_tables.keys()),
        "tables_excluded":       [t for t in allowed_tables if t not in filtered_tables],
        "columns_per_table":     {t: list(cols.keys()) for t, cols in filtered_tables.items()},
        "draft_prompt_injected": query_type == "draft",
        "semantic_cache_active": SEMANTIC_AVAILABLE and hasattr(app.state, "embedder"),
        "max_retries":           MAX_RETRIES,
    }


# ============================
# LIFESPAN INIT
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("FastAPI Lifespan Started")
    logger.info("=" * 60)

    try:
        engine, llm = create_agent()
        await check_ollama()
        await warmup_model()

        app.state.engine = engine
        app.state.llm    = llm

        table_columns = {}

        with engine.connect() as conn:
            logger.info("Connected to database, loading schema...")

            tables = conn.execute(text("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA='dw'
            """)).fetchall()

            logger.info(f"Found {len(tables)} tables in dw schema")

            for (t,) in tables:
                full = f"dw.{t}".lower()
                if full in EXCLUDE_TABLES:
                    logger.debug(f"Excluding table: {full}")
                    continue

                table_columns[full] = {}
                cols = conn.execute(text(f"""
                    SELECT COLUMN_NAME, DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA='dw'
                    AND TABLE_NAME='{t}'
                """)).fetchall()

                logger.debug(f"Loaded {len(cols)} columns for table {full}")

                for col_name, data_type in cols:
                    table_columns[full][col_name.lower()] = data_type.lower()

        app.state.table_columns = table_columns
        logger.info(f"Schema loaded: {len(table_columns)} tables available")

        app.state.metadata = load_metadata()

        example_count = len(
            app.state.metadata.get("examples", {}).get("examples", [])
        )
        if example_count == 0:
            msg = (
                "WARNING: 0 examples loaded from metadata/examples.json!\n"
                "  ALL questions will be sent to the LLM -- check the file for JSON errors.\n"
                "  Run: python -m json.tool metadata/examples.json\n"
            )
            logger.error(msg)
            print(f"\n*** {msg}***\n")
        else:
            logger.info(f"Metadata loaded: {example_count} examples")
            print(f"    Examples loaded: {example_count}")

        if SEMANTIC_AVAILABLE:
            print("    Building semantic example index (sentence-transformers)...")
            app.state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            _rebuild_embeddings(app.state)
            print(f"    Semantic index ready: {len(getattr(app.state, 'example_list', []))} examples")
        else:
            print("    sentence-transformers not installed -- using SequenceMatcher fallback")
            print("    Install with: pip install sentence-transformers")

        print("\n=== PROMPT BLOCKS LOADED ===")
        for key, block in PROMPT_BLOCKS.items():
            status = "OK" if block.strip() else "*** EMPTY ***"
            print(f"  {key:15s} {len(block):5d} chars  {status}")

        print("\n=== METADATA FILES LOADED ===")
        for key, val in app.state.metadata.items():
            if key == "examples":
                count = len(val.get("examples", []))
                print(f"  {key:20s} {count} examples")
            else:
                print(f"  {key:20s} loaded OK")

        print("\n=== ORDERING RULES ===")
        print(f"  ORDERING_RULES: {len(ORDERING_RULES)} chars  {'OK' if ORDERING_RULES.strip() else '*** EMPTY ***'}")
        print("=" * 30 + "\n")

    except Exception as e:
        logger.error(f"ERROR during lifespan init: {e}", exc_info=True)
        raise

    yield

    logger.info("FastAPI Lifespan Shutdown")


app = FastAPI(lifespan=lifespan)


# ============================
# TIMEOUT MIDDLEWARE  (TO-1)
# ============================

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=150.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "Request timed out after 150s -- LLM may be overloaded or routing is wrong"},
            status_code=504
        )


# ============================
# SQL HELPERS
# ============================

WRONG_SCHEMA_PATTERNS = [
    (re.compile(r'\bdr\.(\w)', re.IGNORECASE), r'dw.\1'),
    (re.compile(r'\bdb\.(\w)', re.IGNORECASE), r'dw.\1'),
    (re.compile(r'\bdbo\.(\w)', re.IGNORECASE), r'dw.\1'),
]


def fix_wrong_schema(sql: str) -> str:
    for pattern, replacement in WRONG_SCHEMA_PATTERNS:
        corrected = pattern.sub(replacement, sql)
        if corrected != sql:
            logger.warning(f"Schema prefix auto-corrected: '{pattern.pattern}' -> '{replacement}'")
            sql = corrected
    return sql


def extract_select(raw: str) -> str:
    raw = raw.replace("```sql", "").replace("```", "")

    match = SELECT_REGEX.search(raw)
    if not match:
        raise ValueError("No SELECT")

    sql = raw[match.start():]

    if ";" in sql:
        sql = sql.split(";")[0]

    lines = sql.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(Note|This|I |The |Here|Please|Also|Finally|So |In )', stripped):
            break
        clean_lines.append(line)

    sql = "\n".join(clean_lines).strip()
    sql = fix_wrong_schema(sql)

    sql = re.sub(r'\s+NULLS\s+(?:LAST|FIRST)\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bILIKE\b',               'LIKE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+LIMIT\s+\d+\b',        '',     sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bWITH\s+TIES\b', '', sql, flags=re.IGNORECASE)

    return sql


def enforce_top(sql: str) -> str:
    sql = re.sub(
        r'\bSELECT\s+(DISTINCT\s+)?TOP\s*\(\s*\d+\s*\)\s*',
        lambda m: f"SELECT {(m.group(1) or '').strip()} ".lstrip(),
        sql,
        flags=re.IGNORECASE
    )

    sql_stripped = sql.strip()
    if re.match(r'select\s+count\s*\(\s*\*\s*\)', sql_stripped, re.IGNORECASE) and \
       'group by' not in sql_stripped.lower():
        return sql

    return re.sub(
        r'\bSELECT\b',
        f'SELECT TOP ({AUTO_TOP_LIMIT})',
        sql,
        count=1,
        flags=re.IGNORECASE
    )


def block_injection(sql: str):
    lower = sql.lower()
    for bad in FORBIDDEN:
        if bad in lower:
            raise ValueError("Forbidden SQL")


def extract_alias_map(sql: str):
    alias_map = {}
    for table, alias in TABLE_ALIAS_REGEX.findall(sql):
        alias_map[alias.lower()] = table.lower()
    return alias_map


def validate_columns(sql: str, allowed_tables: dict):
    alias_map = extract_alias_map(sql)

    for full_table in allowed_tables:
        short = full_table.split(".")[-1].lower()
        if short not in alias_map:
            alias_map[short] = full_table.lower()

    allowed_lower = {k.lower(): k for k in allowed_tables}
    for referenced_table in re.findall(r'\bdw\.\w+', sql, re.IGNORECASE):
        if referenced_table.lower() not in allowed_lower:
            valid = ", ".join(allowed_tables.keys())
            raise ValueError(
                f"Table '{referenced_table}' does not exist. "
                f"Valid tables are: {valid}"
            )

    all_short_names  = {t.split(".")[-1].lower() for t in allowed_tables}
    all_schema_names = {t.split(".")[0].lower()  for t in allowed_tables}

    for match in re.finditer(r'\b(?:from|join)\s+([\w]+(?:\.[\w]+)?)\b', sql, re.IGNORECASE):
        ref = match.group(1).lower()
        if "." in ref:              continue
        if ref in all_schema_names: continue
        if ref in all_short_names:  continue
        if ref in SQL_RESERVED_WORDS: continue
        if ref in alias_map:        continue
        raise ValueError(
            f"Table '{ref}' does not exist. "
            f"Valid tables are: {', '.join(allowed_tables.keys())}"
        )

    for alias, column in COLUMN_REGEX.findall(sql):
        alias  = alias.lower()
        column = column.lower()
        if alias in alias_map:
            table = alias_map[alias]
        elif f"dw.{alias}" in allowed_tables:
            table = f"dw.{alias}"
        else:
            continue
        if table not in allowed_tables:
            continue
        if column not in allowed_tables[table]:
            valid_cols = ", ".join(allowed_tables[table].keys())
            raise ValueError(
                f"Invalid column '{column}' on {table}. "
                f"Valid columns are: {valid_cols}"
            )


def fix_invalid_aggregates(sql: str) -> str:
    s = sql.lower()
    has_top       = "top (" in s
    has_aggregate = any(fn in s for fn in ["count(", "sum(", "avg(", "min(", "max("])

    if has_top and has_aggregate:
        sql = re.sub(
            r"select\s+top\s*\(\s*\d+\s*\)\s*",
            "SELECT ",
            sql,
            flags=re.IGNORECASE
        )

    return sql


# ============================
# TABLE FILTERING
# ============================

def filter_relevant_tables(question, allowed_tables):
    keywords = re.findall(r'\b\w+\b', question.lower())

    is_game_record = bool(re.search(
        r'\brecord\s+against\b|\bbest\s+record\s+against\b|\bopponent\b'
        r'|\bhead.to.head\b|\ball.time\b|\bhistorical\s+record\b'
        r'|\bsince\s+\d{4}\b|\bworst\s+record\s+against\b',
        question, re.IGNORECASE
    ))

    is_roster = bool(re.search(
        r'\broster\b|\bjersey.number\b|\bjersey\b|\bwears\b|\bon.the.team\b'
        r'|\breleased\b|\bwaived\b|\btraded\b|\bdesignated.for.assignment\b|\bdfa\b'
        r'|\binjured.list\b|\b10.day\b|\b15.day\b|\b60.day\b|\bil\b'
        r'|\bbereavement\b|\bpaternity\b|\bsuspended\b|\brestricted.list\b'
        r'|\breassigned\b|\bminor.league\b|\bminors\b|\bfree.agent\b'
        r'|\bclaimed\b|\bwaiver\b|\bforty.man\b|\b40.man\b'
        r'|\bnon.roster\b|\bvoluntarily.retired\b|\badministrative.leave\b'
        r'|\btemporary.inactive\b|\bineligible.list\b|\breserve.list\b',
        question, re.IGNORECASE))
    if is_roster:
        filtered = {t: allowed_tables[t] for t in ROSTER_TABLES if t in allowed_tables}
        logger.debug(f"Roster query -- restricting to: {list(filtered.keys())}")
        return filtered

    is_draft = bool(re.search(r'\bdraft', question, re.IGNORECASE))
    if is_draft:
        filtered = {t: allowed_tables[t] for t in DRAFT_TABLES if t in allowed_tables}
        logger.debug(f"Draft query -- restricting to: {list(filtered.keys())}")
        return filtered

    # TO-5: added hitter.season and season.stats
    is_hitter = bool(re.search(
        r'\bhitter\b|\bexit.velocit|\bbarrel\b|\bwrc\b|\blaunch.angle\b'
        r'|\bhard.hit\b|\bcontact.rate\b|\bsprint.speed\b|\biso\b'
        r'|\bhitter_season\b|\bhitter.season\b|\bseason.stats\b',
        question, re.IGNORECASE))
    if is_hitter:
        hitter_only = HITTER_TABLES + ALWAYS_INCLUDE_TABLES
        filtered = {t: allowed_tables[t] for t in hitter_only if t in allowed_tables}
        logger.debug(f"Hitter query -- restricting to: {list(filtered.keys())}")
        return filtered

    is_player_stats = bool(re.search(
        r'\bhome.run\b|\bbatting.average\b|\bera\b|\bwhip\b|\brbi\b'
        r'|\bstolen.base\b|\bops\b|\bstarter\b|\breliever\b'
        r'|\bplayer.stats\b|\bpitching.stat\b|\bhitting.stat\b'
        r'|\bstat__\b|\bleader\b|\bsaves\b|\bshutout\b|\bcomplete.game\b'
        r'|\bwild.pitch\b|\bfielding.percentage\b|\bstrikeout.leader\b'
        r'|\bmost.wins\b|\bwin.leader\b|\bpitcher.*wins\b|\bwins.*pitcher\b'
        r'|\bmost wins\b|\bpitchers.*wins\b',
        question, re.IGNORECASE))
    if is_player_stats:
        filtered = {t: allowed_tables[t] for t in PLAYER_STATS_TABLES if t in allowed_tables}
        logger.debug(f"Player stats query -- restricting to: {list(filtered.keys())}")
        return filtered

    is_zone = bool(re.search(
        r'\bzone\b|\bpitch.location\b|\bheart.of.the.plate\b|\bchase\b'
        r'|\bpitcher_zone\b|\bin.zone\b|\bout.of.zone\b',
        question, re.IGNORECASE))
    if is_zone:
        zone_only = ZONE_TABLES + ALWAYS_INCLUDE_TABLES
        filtered = {t: allowed_tables[t] for t in zone_only if t in allowed_tables}
        logger.debug(f"Zone query -- restricting to: {list(filtered.keys())}")
        return filtered

    is_pitcher_features = bool(re.search(
        r'\bspin.rate\b|\bwhiff.rate\b|\bstrike.rate\b|\busage.pct\b'
        r'|\bpitch.arsenal\b|\bpitcher_pitch_type\b',
        question, re.IGNORECASE))
    if is_pitcher_features:
        pitcher_only = PITCHER_TABLES + ALWAYS_INCLUDE_TABLES
        filtered = {t: allowed_tables[t] for t in pitcher_only if t in allowed_tables}
        logger.debug(f"Pitcher feature query -- restricting to: {list(filtered.keys())}")
        return filtered

    is_atbat = bool(re.search(
        r'\bat.bat\b|\batbat\b|\bfact_atbat\b|\bfull.count\b|\bpitches.per\b',
        question, re.IGNORECASE))
    if is_atbat:
        filtered = {t: allowed_tables[t] for t in ATBAT_TABLES if t in allowed_tables}
        logger.debug(f"At-bat query -- restricting to: {list(filtered.keys())}")
        return filtered

    # TO-2: Opening Day routes to GAME_TABLES (not SEASON_TABLES)
    is_opening_day = bool(re.search(r'\bopening.day\b', question, re.IGNORECASE))
    if is_opening_day:
        filtered = {t: allowed_tables[t] for t in GAME_TABLES if t in allowed_tables}
        logger.debug(f"Opening Day query -- restricting to: {list(filtered.keys())}")
        return filtered

    # TO-3: Pitch core routing for hits/pitch_type/batter_side queries
    is_pitch_core = bool(re.search(
        r'\bpitch.type\b|\bpitcher.hand\b|\bbatter.side\b|\bp_throws\b'
        r'|\bpitch_core\b|\bhits\b.*\bpitch\b|\bpitch\b.*\bhits\b'
        r'|\bevents\b.*\bpitch\b|\bpitch\b.*\bevents\b',
        question, re.IGNORECASE))
    if is_pitch_core:
        filtered = {t: allowed_tables[t] for t in PITCH_CORE_TABLES if t in allowed_tables}
        logger.debug(f"Pitch core query -- restricting to: {list(filtered.keys())}")
        return filtered

    # NOTE: \bopening.day\b intentionally removed from is_season (TO-2)
    is_season = bool(re.search(
        r'\bseason.dates\b|\bspring.training\b|\ball.star.date\b'
        r'|\bpostseason.start\b|\boffseason\b|\bqualifier\b',
        question, re.IGNORECASE))
    if is_season:
        filtered = {t: allowed_tables[t] for t in SEASON_TABLES if t in allowed_tables}
        logger.debug(f"Season query -- restricting to: {list(filtered.keys())}")
        return filtered

    if is_game_record:
        filtered = {t: allowed_tables[t] for t in GAME_TABLES if t in allowed_tables}
        logger.debug(f"Game record query -- restricting to: {list(filtered.keys())}")
        return filtered

    filtered = {}
    for table, cols in allowed_tables.items():
        table_short = table.split(".")[-1].lower()
        if any(k == table_short or k in table_short.split("_") for k in keywords):
            filtered[table] = cols

    for t in ALWAYS_INCLUDE_TABLES:
        if t in allowed_tables and t not in filtered:
            filtered[t] = allowed_tables[t]

    return filtered if filtered else allowed_tables


def fix_draft_team_join(sql: str) -> str:
    if not re.search(r'\bdw\.draft\b', sql, re.IGNORECASE):
        return sql
    if not re.search(r'\bjoin\s+dw\.teams\b', sql, re.IGNORECASE):
        return sql

    alias_match = re.search(
        r'\bjoin\s+dw\.teams\s+(?:as\s+)?(\w+)', sql, re.IGNORECASE)
    if alias_match:
        teams_alias = alias_match.group(1)
        sql = re.sub(
            rf'\b{re.escape(teams_alias)}\.team__name\b',
            'd.team__name',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            rf'\b{re.escape(teams_alias)}\.name\b',
            'd.team__name',
            sql,
            flags=re.IGNORECASE
        )

    sql = re.sub(
        r'\s*(?:left\s+|inner\s+)?join\s+dw\.teams\s+(?:as\s+)?\w+\s+on\s+\S+\s*=\s*\S+',
        '',
        sql,
        flags=re.IGNORECASE
    )

    logger.warning("fix_draft_team_join: auto-removed illegal JOIN dw.teams and replaced with d.team__name")
    return sql


def fix_fake_team_id_joins(sql: str) -> str:
    if not re.search(r'teams__(?:home|away)__team__id', sql, re.IGNORECASE):
        return sql

    sql = re.sub(r'\bt_home\.name\b', 'teams__home__team__name', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bt_away\.name\b', 'teams__away__team__name', sql, flags=re.IGNORECASE)

    sql = re.sub(
        r'\s*(?:left\s+|inner\s+|right\s+)?join\s+dw\.teams\s+(?:as\s+)?t_(?:home|away)'
        r'\s+on\s+g\.teams__(?:home|away)__team__id\s*=\s*t_(?:home|away)\.id',
        '',
        sql,
        flags=re.IGNORECASE
    )

    logger.warning("fix_fake_team_id_joins: auto-removed fake team ID JOINs, replaced with flat columns")
    return sql


# ============================
# SQL GENERATION
# ============================

def generate_sql(question, llm, allowed_tables, metadata):
    logger.info(f"QUESTION: {question}")

    has_year_range = bool(re.search(r'\b\d{4}\s*(?:-|to)\s*\d{4}\b', question, re.IGNORECASE))
    logger.info(f"has_year_range={has_year_range} for question: {question}")

    # Example cache lookup
    matched_sql, match_score = find_best_example(question, metadata, threshold=0.99)
    if matched_sql:
        if has_year_range:
            q_range = re.search(r'\b(\d{4})\s*(?:-|to)\s*(\d{4})\b', question, re.IGNORECASE)
            if q_range:
                y1, y2 = q_range.group(1), q_range.group(2)
                if sql_matches_question_context(matched_sql, question, y1, y2):
                    logger.info(f"Using example SQL with matching year range and context (score: {match_score:.0%})")
                    return matched_sql, [], "example"
                else:
                    logger.info(
                        f"Example match ({match_score:.0%}) skipped -- "
                        f"year range ({y1}-{y2}) or team name not confirmed in cached SQL, routing to LLM"
                    )
        else:
            question_lower = question.lower()
            team_in_question = next((t for t in MLB_TEAMS if t in question_lower), None)
            if team_in_question and team_in_question not in matched_sql.lower():
                logger.info(
                    f"Example match ({match_score:.0%}) skipped -- "
                    f"team '{team_in_question}' not found in cached SQL, routing to LLM"
                )
            else:
                years_in_question = re.findall(r'\b(20\d{2})\b', question)
                years_mismatch = any(
                    not sql_contains_year(matched_sql, y) for y in years_in_question
                )
                if years_mismatch:
                    logger.info(
                        f"Example match ({match_score:.0%}) skipped -- "
                        f"year(s) {years_in_question} not confirmed in cached SQL, routing to LLM"
                    )
                else:
                    logger.info(f"Using example SQL (match score: {match_score:.0%})")
                    return matched_sql, [], "example"

    # Enrich + filter tables
    question_enriched = enrich_question(question)
    filtered_tables   = filter_relevant_tables(question_enriched, allowed_tables)
    logger.info(f"Tables sent to LLM: {list(filtered_tables.keys())}")

    query_type   = get_query_type(question)
    domain_block = PROMPT_BLOCKS.get(query_type, "")
    logger.info(f"Query type: {query_type}")

    WIDE_TABLES = {
        "dw.draft", "dw.hitter_season_features",
        "dw.pitcher_pitch_type_features", "dw.pitch_core",
        "dw.pitcher_zone_counts", "dw.fact_atbat",
        "dw.player_stats"
    }
    schema_lines = []
    for t, cols in filtered_tables.items():
        cap = 120 if t in WIDE_TABLES else 80
        relevant_cols = filter_columns_for_question(question_enriched, t, cols, cap)
        col_str = ", ".join(
            f"{col}({dtype})" for col, dtype in relevant_cols.items()
        )
        schema_lines.append(f"{t}: {col_str}")
    schema_text = "\n".join(schema_lines)

    logger.debug(f"Schema sent to LLM:\n{schema_text}")

    if query_type == "roster":
        metadata_text = ""
    else:
        metadata_text = format_metadata_for_prompt(metadata)
        metadata_text = trim_to_token_budget(metadata_text, max_chars=2000)

    schema_text = trim_to_token_budget(schema_text, max_chars=8000)

    valid_tables_list = ", ".join(filtered_tables.keys())

    base_prompt = f"""Generate a valid Microsoft SQL Server (T-SQL) SELECT statement ONLY.

CRITICAL SCHEMA RULE:
- Schema is EXACTLY "dw" -- EVERY table MUST start with "dw."
- NEVER write "dr.", "db.", "dbo." -- ONLY "dw."

{ORDERING_RULES}

Rules:
- T-SQL / Microsoft SQL Server syntax ONLY.
- FORBIDDEN: to_char, to_date, to_number, date_trunc, extract, NULLS FIRST, NULLS LAST, ILIKE, LIMIT, ::
- Use: FORMAT(), CONVERT(), CAST(), TRY_CAST(), YEAR(), MONTH(), DAY(), ISNULL(), TOP(N)
- ONLY use tables listed in Valid Tables.
- Valid tables: {valid_tables_list}

{domain_block}

{metadata_text}

Schema:
{schema_text}

Question: {question_enriched}

Return ONLY raw T-SQL. No explanation. No markdown. End with semicolon."""

    draft_join_keywords      = ["must NOT join dw.teams"]
    invalid_col_keywords     = ["is_home_run", "original_game_date", "Invalid column", "ORDER BY game_date"]
    groupby_gamepk_keywords  = ["GROUP BY game_pk"]
    fake_team_id_keywords    = ["FAKE_TEAM_ID_JOIN"]
    juco_filter_keywords     = ["JC_FILTER_MISSING"]

    attempts = []
    for i in range(MAX_RETRIES):
        try:
            if attempts:
                last_error = attempts[-1]

                if "dr." in last_error:
                    error_hint = (
                        f"\n\nCRITICAL: Wrong schema prefix. {last_error[:200]}"
                        f"\nReplace ALL 'dr.' with 'dw.' -- schema is dw. ONLY."
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif "PostgreSQL" in last_error or "Invalid SQL Server syntax" in last_error:
                    error_hint = (
                        f"\n\nFAILED -- POSTGRESQL SYNTAX DETECTED: {last_error[:200]}"
                        f"\nYou MUST use T-SQL (Microsoft SQL Server) only."
                        f"\nBANNED: to_char(), to_number(), to_date(), EXTRACT(), ILIKE, LIMIT, :: cast, NULLS FIRST"
                        f"\nUSE INSTEAD: CAST(), TRY_CAST(), FORMAT(), CONVERT(), YEAR(), MONTH(), DAY(), TOP(N)"
                        f"\n\nFor the roster question, the correct SQL is:"
                        f"\n  SELECT TOP (100) r.person__full_name, t.name AS team_name,"
                        f"\n                   r.position__name, r.status__description, r.season"
                        f"\n  FROM dw.rosters r"
                        f"\n  JOIN dw.teams t ON t.id = r.parent_team_id"
                        f"\n  WHERE r.season = 2025"
                        f"\n    AND r.status__description = 'Reassigned to Minors'"
                        f"\n  ORDER BY t.name, r.person__full_name;"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in juco_filter_keywords):
                    error_hint = (
                        f"\n\nFAILED -- MISSING SCHOOL TYPE FILTER: {last_error[:300]}"
                        f"\nThe question asks for junior college draft picks."
                        f"\nYou MUST join dw.school_type_lookup and filter on school_type."
                        f"\nNEVER filter home__state to answer a school type question."
                        f"\n"
                        f"\nCORRECT pattern:"
                        f"\n  SELECT TOP (100) d.person__full_name, d.person__primary_position__name,"
                        f"\n                   d.school__name, s.school_type"
                        f"\n  FROM dw.draft d"
                        f"\n  JOIN dw.school_type_lookup s ON s.school_name = d.school__name"
                        f"\n  WHERE d.year = 2025"
                        f"\n    AND s.school_type IN ('JC', 'Junior College', 'JUCO', 'Community College')"
                        f"\n  ORDER BY d.pick_number;"
                        f"\n"
                        f"\nschool_type values: 'HS' = High School, 'JC' = Junior/Community College, '4-Year' = College"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in draft_join_keywords):
                    error_hint = (
                        f"\n\nFAILED -- ILLEGAL JOIN ON DRAFT QUERY: {last_error[:300]}"
                        f"\ndw.draft is a FLAT table. team__name is already a column on dw.draft."
                        f"\nDo NOT join dw.teams. Remove the JOIN entirely."
                        f"\nCorrect pattern:"
                        f"\n  SELECT d.person__full_name, d.person__primary_position__name, d.team__name"
                        f"\n  FROM dw.draft d"
                        f"\n  WHERE d.year BETWEEN 2020 AND 2025"
                        f"\n    AND d.home__state IN ('CA', 'California')"
                        f"\n  ORDER BY d.year, d.pick_number;"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in invalid_col_keywords):
                    error_hint = (
                        f"\n\nFAILED -- INVALID COLUMN OR ORDER BY: {last_error[:300]}"
                        f"\nFor home runs in dw.pitch_core: use events = 'home_run' (NOT is_home_run)"
                        f"\nFor rescheduled games: use rescheduled_from_date (NOT original_game_date)"
                        f"\nFor postseason GROUP BY queries: use ORDER BY MIN(game_date) (NOT ORDER BY game_date)"
                        f"\nOnly use columns shown in the Schema section."
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in roster_hallucination_keywords) or \
                        ("dw.games" in last_error and "dw.rosters" in last_error.lower()):
                    error_hint = (
                        f"\n\nFAILED -- HALLUCINATED TABLE ON ROSTER QUERY: {last_error[:300]}"
                        f"\nUse ONLY dw.rosters and dw.teams. Do NOT join dw.games."
                        f"\nBanned tables: player_events, transactions, roster_moves."
                        f"\nBanned columns: status_date, event_type_code, transaction_date."
                        f"\nCorrect pattern:"
                        f"\n  SELECT TOP (100) r.person__full_name, r.season, r.status__description"
                        f"\n  FROM dw.rosters r"
                        f"\n  JOIN dw.teams t ON t.id = r.parent_team_id"
                        f"\n  WHERE t.name = 'San Francisco Giants'"
                        f"\n    AND r.season BETWEEN 2024 AND 2025"
                        f"\n    AND r.status__description IN ('Released', 'Waived', 'Traded')"
                        f"\n  ORDER BY r.season DESC, r.status__description, r.person__full_name;"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in fake_team_id_keywords):
                    error_hint = (
                        f"\n\nFAILED -- FAKE JOIN COLUMN: {last_error[:300]}"
                        f"\nteams__home__team__id and teams__away__team__id DO NOT EXIST on dw.games."
                        f"\ndw.games has FLAT team name columns -- use them DIRECTLY, no JOIN needed:"
                        f"\n  teams__home__team__name  (use directly in WHERE and SELECT)"
                        f"\n  teams__away__team__name  (use directly in WHERE and SELECT)"
                        f"\n"
                        f"\nCORRECT pattern -- NO JOIN to dw.teams:"
                        f"\n  SELECT TOP (100)"
                        f"\n      teams__away__team__name                                          AS opponent,"
                        f"\n      SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)      AS wins,"
                        f"\n      SUM(CASE WHEN teams__away__is_winner = 1 THEN 1 ELSE 0 END)      AS losses,"
                        f"\n      COUNT(*)                                                          AS games_played,"
                        f"\n      CAST("
                        f"\n          SUM(CASE WHEN teams__home__is_winner = 1 THEN 1 ELSE 0 END)"
                        f"\n          * 1.0 / NULLIF(COUNT(*), 0)"
                        f"\n      AS decimal(5,3))                                                  AS win_pct"
                        f"\n  FROM dw.games"
                        f"\n  WHERE season BETWEEN '2020' AND '2025'"
                        f"\n    AND series_description = 'Regular Season'"
                        f"\n    AND status__detailed_state = 'Final'"
                        f"\n    AND teams__home__team__name = 'San Francisco Giants'"
                        f"\n  GROUP BY teams__away__team__name"
                        f"\n  ORDER BY win_pct ASC;"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif any(k in last_error for k in groupby_gamepk_keywords) or \
                     re.search(r'group\s+by\s+\w+\.game_pk', last_error, re.IGNORECASE):
                    error_hint = (
                        f"\n\nFAILED -- WRONG GROUP BY OR BANNED JOIN COLUMNS: {last_error[:300]}"
                        f"\nFor home record queries:"
                        f"\n  - GROUP BY must be the TEAM NAME column, NOT game_pk"
                        f"\n  - NEVER join dw.teams using teams__home__team__id -- that column does not exist"
                        f"\n  - Use flat columns teams__home__team__name / teams__away__team__name DIRECTLY on dw.games"
                        f"\n  - CAST must include the type: CAST(... AS decimal(5,3))"
                        f"\nCorrect pattern:"
                        f"\n  SELECT g.teams__home__team__name AS home_team,"
                        f"\n         COUNT(*) AS games,"
                        f"\n         SUM(CASE WHEN g.teams__home__is_winner = 1 THEN 1 ELSE 0 END) AS wins,"
                        f"\n         SUM(CASE WHEN g.teams__home__is_winner = 0 THEN 1 ELSE 0 END) AS losses,"
                        f"\n         CAST(SUM(CASE WHEN g.teams__home__is_winner = 1 THEN 1 ELSE 0 END)"
                        f"\n              * 1.0 / NULLIF(COUNT(*), 0) AS decimal(5,3)) AS win_pct"
                        f"\n  FROM dw.games g"
                        f"\n  WHERE g.teams__home__team__name = 'San Francisco Giants'"
                        f"\n    AND g.season BETWEEN '2020' AND '2025'"
                        f"\n    AND g.series_description = 'Regular Season'"
                        f"\n    AND g.status__detailed_state = 'Final'"
                        f"\n  GROUP BY g.teams__home__team__name"
                        f"\n  ORDER BY win_pct DESC;"
                        f"\nReturn ONLY the corrected SQL."
                    )

                elif "worst/lowest/fewest" in last_error or "best/highest/most" in last_error:
                    question_lower = question.lower()
                    is_worst = any(kw in question_lower for kw in [
                        "worst", "lowest", "fewest", "least", "bottom", "most losses"
                    ])
                    error_hint = (
                        f"\n\nFAILED -- WRONG ORDER BY DIRECTION: {last_error[:200]}"
                        f"\nORDERING RULES:"
                        f"\n  'worst', 'lowest', 'fewest', 'least'  → ORDER BY <metric> ASC"
                        f"\n  'best',  'highest', 'most',  'top'    → ORDER BY <metric> DESC"
                        f"\n"
                        f"\nFor 'worst home record': ORDER BY win_pct ASC"
                        f"\nFor 'best home record':  ORDER BY win_pct DESC"
                        f"\n"
                        f"\nThis question asks for '{'worst' if is_worst else 'best'}' -- "
                        f"use ORDER BY win_pct {'ASC' if is_worst else 'DESC'}."
                        f"\nReturn ONLY the corrected SQL with the fixed ORDER BY direction."
                    )

                else:
                    error_hint = (
                        f"\n\nFAILED: {last_error[:200]}"
                        f"\nUse ONLY these tables: {valid_tables_list}"
                        f"\nAll tables must start with 'dw.'"
                        f"\nReturn ONLY the corrected SQL."
                    )

                prompt = base_prompt + error_hint
            else:
                prompt = base_prompt

            logger.info(f"Ollama attempt {i + 1}/{MAX_RETRIES}")
            start_llm = time.time()
            raw = llm.invoke(prompt)
            llm_time = round(time.time() - start_llm, 2)
            logger.debug(f"Ollama response time: {llm_time}s\nRaw:\n{raw[:500]}")

            sql = extract_select(raw)
            sql = enforce_top(sql)
            sql = fix_invalid_aggregates(sql)
            sql = fix_draft_team_join(sql)
            sql = fix_fake_team_id_joins(sql)
            sql = fix_ambiguous_columns(sql)

            block_injection(sql)
            validate_tsql(sql)
            validate_draft_no_join(sql)
            validate_school_type_filter(sql, question)
            validate_columns(sql, allowed_tables)
            validate_groupby(sql)
            validate_no_game_pk_in_group_by(sql)
            validate_cast_has_type(sql)
            validate_no_fake_team_id_join(sql)
            validate_ordering_direction(sql, question)

            logger.info(f"SQL generated on attempt {i + 1}")
            return sql, attempts, "llm"

        except Exception as e:
            error_msg = str(e)
            attempts.append(error_msg)
            logger.warning(f"Attempt {i + 1} failed: {error_msg}")

    error_summary = " | ".join(attempts)
    logger.error(f"All {MAX_RETRIES} attempts failed: {error_summary}")
    raise ValueError(attempts)


# ============================
# METADATA
# ============================

def load_metadata():
    if not os.path.exists(METADATA_FOLDER):
        logger.warning(f"Metadata folder not found: {METADATA_FOLDER}")
        return {}

    metadata = {}
    for file in os.listdir(METADATA_FOLDER):
        if file.endswith(".json"):
            path = os.path.join(METADATA_FOLDER, file)
            try:
                with open(path, encoding="utf-8") as f:
                    key = file.replace(".json", "")
                    metadata[key] = json.load(f)
                logger.info(f"Loaded metadata: {file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file}: {e}")
                print(f"\n*** JSON PARSE ERROR in metadata/{file} ***\n"
                      f"    {e}\n"
                      f"    Fix: python -m json.tool metadata/{file}\n")
            except Exception as e:
                logger.error(f"Could not load {file}: {e}")

    return metadata


def format_metadata_for_prompt(metadata: dict) -> str:
    sections = []

    if "rules" in metadata:
        sections.append("\nQUERY RULES:")
        for rule in metadata["rules"].get("rules", []):
            sections.append(f"  - {rule}")
        sections.append("  NEVER DO:")
        for mistake in metadata["rules"].get("common_mistakes", []):
            sections.append(f"  - {mistake}")

    if "relationships" in metadata:
        sections.append("\nCORRECT JOIN COLUMNS:")
        for j in metadata["relationships"].get("joins", []):
            label = j.get("label", "")
            sections.append(
                f"  {j['from_table']}.{j['from_column']} -> "
                f"{j['to_table']}.{j['to_column']}"
                + (f" ({label})" if label else "")
            )
        for n in metadata["relationships"].get("never_join_on", []):
            sections.append(
                f"  NEVER JOIN ON: {n.get('pattern', '')} -- {n.get('reason', '')}")

    if "columns" in metadata:
        sections.append("\nCOLUMN NOTES:")
        for table, cols in metadata["columns"].items():
            for col, note in cols.items():
                sections.append(f"  {table}.{col}: {note}")

    if "tables" in metadata:
        sections.append("\nTABLE DESCRIPTIONS:")
        for table, info in metadata["tables"].items():
            desc = info.get("description", "") if isinstance(info, dict) else info
            sections.append(f"  {table}: {desc}")

    return "\n".join(sections)


# ============================
# RESULT VALIDATION
# ============================

def validate_results(rows: list, question: str) -> list:
    warnings = []
    if len(rows) == 0:
        warnings.append("Query returned 0 rows -- possible over-filtering or wrong season filter")
    if len(rows) >= AUTO_TOP_LIMIT:
        warnings.append(f"Result hit TOP({AUTO_TOP_LIMIT}) limit -- results may be truncated")
    if rows:
        col_count = len(rows[0])
        for col_idx in range(col_count):
            if all(r[col_idx] is None for r in rows):
                warnings.append(
                    f"Column index {col_idx} is entirely NULL -- "
                    f"possible wrong column name or missing JOIN"
                )
    return warnings


# ============================
# ENDPOINTS
# ============================

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head><title>MLB Query</title></head>
    <body style="font-family: Arial; max-width: 400px; margin: 60px auto; text-align: center;">
        <h2>&#9918; MLB AI Query</h2>
        <a href="/ask"      style="display:block; margin:10px; padding:12px; background:#1a1a2e; color:white; text-decoration:none; border-radius:6px;">Query Interface</a>
        <a href="/reflect"  style="display:block; margin:10px; padding:12px; background:#2d6b2d; color:white; text-decoration:none; border-radius:6px;">Reflection Dashboard</a>
        <a href="/examples" style="display:block; margin:10px; padding:12px; background:#444; color:white; text-decoration:none; border-radius:6px;">View Examples</a>
        <a href="/model"    style="display:block; margin:10px; padding:12px; background:#444; color:white; text-decoration:none; border-radius:6px;">Model Info</a>
        <a href="/docs"     style="display:block; margin:10px; padding:12px; background:#444; color:white; text-decoration:none; border-radius:6px;">API Docs</a>
    </body>
    </html>
    """


@app.get("/ask", response_class=HTMLResponse)
def ask_form():
    return """
    <html>
    <head>
        <title>MLB Query</title>
        <style>
            body { font-family: Arial; max-width: 900px; margin: 40px auto; padding: 20px; background: #f9f9f9; }
            h2 { color: #1a1a2e; }
            textarea { width: 100%; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
            button { margin-top: 10px; padding: 10px 28px; font-size: 15px; background: #1a1a2e; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #2d2d6b; }
            label { font-weight: bold; display: block; margin-top: 20px; margin-bottom: 6px; color: #333; }
            pre { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; min-height: 40px; }
            #meta { font-size: 13px; color: #666; margin-top: 8px; }
            #warnings { color: #b36200; font-size: 13px; margin-top: 6px; }
            #error { color: red; font-weight: bold; margin-top: 10px; }
            #spinner { display: none; margin-top: 10px; color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <h2>&#9918; MLB AI Query</h2>
        <label for="question">Question</label>
        <textarea id="question" rows="3" placeholder="e.g. top 20 home run hitters in 2025"></textarea>
        <button id="askBtn">Run Query</button>
        <div id="spinner">Thinking...</div>
        <div id="error"></div>
        <div id="meta"></div>
        <div id="warnings"></div>
        <label>SQL</label>
        <pre id="sql"></pre>
        <label>Results</label>
        <pre id="answer"></pre>
        <script>
        async function ask() {
            const question = document.getElementById("question").value.trim();
            if (!question) { alert("Please enter a question"); return; }
            document.getElementById("sql").textContent = "";
            document.getElementById("answer").textContent = "";
            document.getElementById("meta").textContent = "";
            document.getElementById("warnings").textContent = "";
            document.getElementById("error").textContent = "";
            document.getElementById("spinner").style.display = "block";
            try {
                const res = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({question: question})
                });
                const data = await res.json();
                document.getElementById("spinner").style.display = "none";
                if (data.error) {
                    document.getElementById("error").textContent = "Error: " + data.error;
                } else {
                    document.getElementById("meta").textContent =
                        "Source: " + data.source + "  |  LLM: " + data.time_llm +
                        "s  DB: " + data.time_db + "s  Total: " + data.time_total + "s";
                    if (data.warnings && data.warnings.length > 0) {
                        document.getElementById("warnings").textContent =
                            "⚠ " + data.warnings.join("  |  ⚠ ");
                    }
                    document.getElementById("sql").textContent = data.sql;
                    const rows = data.answer;
                    if (rows.length === 0) {
                        document.getElementById("answer").textContent = "No results.";
                    } else {
                        document.getElementById("answer").textContent =
                            rows.map(r => r.join("  |  ")).join("\\n");
                    }
                }
            } catch (e) {
                document.getElementById("spinner").style.display = "none";
                document.getElementById("error").textContent = "Request failed: " + e.message;
            }
        }
        document.getElementById("askBtn").addEventListener("click", e => { e.preventDefault(); ask(); });
        document.getElementById("question").addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
        });
        </script>
    </body>
    </html>
    """


@app.post("/ask")
def ask(req: QuestionRequest):
    start = time.time()
    try:
        t1 = time.time()
        sql, attempts, source = generate_sql(
            req.question,
            app.state.llm,
            app.state.table_columns,
            app.state.metadata
        )
        llm_time = round(time.time() - t1, 3)

        t2 = time.time()
        with app.state.engine.connect() as conn:
            conn.execute(text(f"SET LOCK_TIMEOUT {QUERY_TIMEOUT_SECONDS * 1000}"))
            db_check = conn.execute(text(
                "SELECT DB_NAME() AS db, COUNT(*) AS roster_rows FROM dw.rosters"
            )).fetchone()
            logger.info(f"DB context: db={db_check[0]}, roster_rows={db_check[1]}")
            rows = conn.execute(text(sql)).fetchall()
        db_time = round(time.time() - t2, 3)

        rows_list = [list(r) for r in rows]

        maybe_save_example(req.question, sql, source, rows_list)

        warnings = validate_results(rows_list, req.question)

        logger.info(f"Query successful. LLM: {llm_time}s, DB: {db_time}s, Rows: {len(rows_list)}")

        return {
            "sql":        sql,
            "source":     source,
            "answer":     rows_list,
            "warnings":   warnings,
            "time_llm":   llm_time,
            "time_db":    db_time,
            "time_total": round(time.time() - start, 3)
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query failed: {error_msg}", exc_info=True)
        return {
            "error":      error_msg,
            "time_total": round(time.time() - start, 3)
        }


@app.get("/reflect", response_class=HTMLResponse)
def reflect_dashboard():
    return """
    <html>
    <head>
        <title>MLB Reflection</title>
        <style>
            body { font-family: Arial; max-width: 960px; margin: 40px auto; padding: 20px; background: #f4fff4; }
            h2 { color: #2d6b2d; }
            h3 { color: #444; margin-top: 30px; }
            textarea { width: 100%; padding: 10px; font-size: 14px; border: 1px solid #aaa; border-radius: 6px; box-sizing: border-box; }
            button { margin-top: 10px; padding: 10px 24px; font-size: 14px; background: #2d6b2d; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #1a4a1a; }
            pre { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; min-height: 40px; max-height: 500px; overflow-y: auto; }
            #spinner { display: none; color: #666; font-style: italic; margin-top: 8px; }
        </style>
    </head>
    <body>
        <h2>Reflection Dashboard</h2>
        <p>Inspect how the system interprets a question <strong>without executing it</strong>.</p>
        <label>Question to reflect on:</label>
        <textarea id="question" rows="2" placeholder="e.g. top 20 home run hitters in 2025"></textarea>
        <button id="reflectBtn">Reflect</button>
        <div id="spinner">Analyzing...</div>
        <h3>Routing Decision</h3>
        <pre id="routing"></pre>
        <h3>Tables Selected</h3>
        <pre id="tables"></pre>
        <h3>Top Example Matches</h3>
        <pre id="examples"></pre>
        <h3>Full Reflection JSON</h3>
        <pre id="full"></pre>
        <hr>
        <h3>System State</h3>
        <button id="stateBtn">Load System State</button>
        <pre id="state"></pre>
        <script>
        async function reflect() {
            const question = document.getElementById("question").value.trim();
            if (!question) { alert("Enter a question"); return; }
            document.getElementById("spinner").style.display = "block";
            ["routing","tables","examples","full"].forEach(id => document.getElementById(id).textContent = "");
            try {
                const res = await fetch("/reflect/query", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({question})
                });
                const d = await res.json();
                document.getElementById("spinner").style.display = "none";
                document.getElementById("routing").textContent =
                    "Query type      : " + d.query_type + "\\n" +
                    "Path            : " + d.path + "\\n" +
                    "Example hit     : " + (d.example_match.found
                        ? "YES  score=" + d.example_match.score
                        : "NO   best score=" + d.example_match.score) + "\\n" +
                    "Draft block     : " + (d.draft_prompt_injected ? "INJECTED" : "not used") + "\\n" +
                    "Semantic cache  : " + (d.semantic_cache_active ? "ACTIVE" : "fallback (SequenceMatcher)") + "\\n\\n" +
                    "Enriched question:\\n  " + d.enriched_question;
                document.getElementById("tables").textContent =
                    "Selected (" + d.tables_selected.length + "):\\n  " +
                    d.tables_selected.join("\\n  ") +
                    "\\n\\nExcluded (" + d.tables_excluded.length + "):\\n  " +
                    (d.tables_excluded.join("\\n  ") || "(none)");
                const scoreLines = d.top_example_scores.map(e =>
                    (e.would_match ? "[MATCH] " : "        ") +
                    e.similarity_score.toFixed(3) + "  " + e.example_question
                ).join("\\n");
                document.getElementById("examples").textContent = scoreLines || "(no examples loaded)";
                document.getElementById("full").textContent = JSON.stringify(d, null, 2);
            } catch(e) {
                document.getElementById("spinner").style.display = "none";
                document.getElementById("routing").textContent = "Error: " + e.message;
            }
        }
        async function loadState() {
            const res = await fetch("/reflect/state");
            const d = await res.json();
            document.getElementById("state").textContent = JSON.stringify(d, null, 2);
        }
        document.getElementById("reflectBtn").addEventListener("click", e => { e.preventDefault(); reflect(); });
        document.getElementById("stateBtn").addEventListener("click",  e => { e.preventDefault(); loadState(); });
        document.getElementById("question").addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); reflect(); }
        });
        </script>
    </body>
    </html>
    """


@app.post("/reflect/query")
def reflect_query(req: ReflectQueryRequest):
    return reflect_classify_question(
        req.question,
        app.state.metadata,
        app.state.table_columns
    )


@app.get("/reflect/tables")
def reflect_tables():
    return {
        "table_count": len(app.state.table_columns),
        "tables": {
            table: {
                "column_count": len(cols),
                "columns": cols
            }
            for table, cols in app.state.table_columns.items()
        }
    }


@app.get("/reflect/examples")
def reflect_examples():
    examples = app.state.metadata.get("examples", {}).get("examples", [])
    return {
        "count":     len(examples),
        "threshold": 0.99,
        "examples":  examples
    }


@app.get("/reflect/state")
def reflect_state():
    return {
        "model": {
            "name":        OLLAMA_MODEL,
            "base_url":    OLLAMA_BASE_URL,
            "num_ctx":     app.state.llm.num_ctx,
            "num_predict": app.state.llm.num_predict,
            "temperature": app.state.llm.temperature,
        },
        "config": {
            "max_retries":            MAX_RETRIES,
            "query_timeout_seconds":  QUERY_TIMEOUT_SECONDS,
            "auto_top_limit":         AUTO_TOP_LIMIT,
            "metadata_folder":        METADATA_FOLDER,
        },
        "schema": {
            "table_count":           len(app.state.table_columns),
            "table_names":           list(app.state.table_columns.keys()),
            "excluded_tables":       EXCLUDE_TABLES,
            "always_include_tables": ALWAYS_INCLUDE_TABLES,
        },
        "metadata": {
            "loaded_files":       list(app.state.metadata.keys()),
            "example_count":      len(
                app.state.metadata.get("examples", {}).get("examples", [])
            ),
        },
        "optimizations": {
            "semantic_cache_active":       SEMANTIC_AVAILABLE and hasattr(app.state, "embedder"),
            "semantic_model":              "all-MiniLM-L6-v2" if SEMANTIC_AVAILABLE else None,
            "semantic_example_count":      len(getattr(app.state, "example_list", [])),
            "prompt_domain_block_active":  True,
            "column_relevance_filter":     True,
            "auto_promote_examples":       True,
        }
    }


@app.get("/examples")
def list_examples():
    examples = app.state.metadata.get("examples", {}).get("examples", [])
    return {
        "count":     len(examples),
        "questions": [ex.get("question", "") for ex in examples]
    }


@app.get("/model")
def model_info():
    return {
        "model":       OLLAMA_MODEL,
        "base_url":    OLLAMA_BASE_URL,
        "num_ctx":     app.state.llm.num_ctx,
        "num_predict": app.state.llm.num_predict,
        "temperature": app.state.llm.temperature,
        "provider":    "ollama"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )