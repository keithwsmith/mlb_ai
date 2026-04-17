# prepare_training_data.py
import json

SCHEMA_CONTEXT = """
Database: SQL Server (T-SQL). Schema dw.
Tables: dw.draft, dw.games, dw.teams, dw.venues, dw.school_type_lookup

dw.draft columns: year, pick_number, pick_round, person__full_name,
  person__primary_position__name, home__state, home__city, team__name,
  school__name, signing_bonus (nvarchar - CAST to decimal for math)

dw.games columns: game_pk, game_date, series_description, double_header,
  status__detailed_state, teams__home__score, teams__away__score,
  teams__home__team__id, teams__away__team__id, teams__home__is_winner

dw.teams columns: id, name, team_name, abbreviation, league__name, division__name

dw.school_type_lookup columns: school_name, school_type (High School/Junior College/University/College)

RULES:
- Use TOP (N) not LIMIT
- Use LIKE not ILIKE
- Use YEAR() not EXTRACT()
- Use CAST() not :: operator
- Never use NULLS FIRST/LAST
- dw.draft.signing_bonus is nvarchar — always CAST(signing_bonus AS decimal(18,2))
- State filter: use IN ('CA', 'California') not = 'CA'
"""

with open("metadata/examples.json") as f:
    examples = json.load(f)["examples"]

training_data = []
for ex in examples:
    training_data.append({
        "instruction": f"Generate a valid Microsoft SQL Server T-SQL SELECT statement only. Output raw SQL only, no explanation.\n\n{SCHEMA_CONTEXT}",
        "input": ex["question"],
        "output": ex["sql"]
    })

with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Created {len(training_data)} training examples")