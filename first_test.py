"""
First test (ChatOpenRouter version): send ONE question to ONE model via
OpenRouter, print the raw SQL. No loop, no scoring, no cleaning.

Setup:
  pip install langchain-openrouter langchain-core python-dotenv
  Then set OPENROUTER_API_KEY in your .env file (or environment).
"""

import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load variables from a .env file in the current directory (if present)
load_dotenv()

# ---------- CONFIG ----------
# OpenRouter model slug. Good cheap options:
#   "openai/gpt-4o-mini"          (cheap, fast, good baseline)
#   "openai/gpt-4o"               (stronger, more expensive)
#   "anthropic/claude-3.5-sonnet" (great at SQL)
#   "google/gemini-2.0-flash-001" (very cheap)
MODEL = "openai/gpt-4o-mini"

# Read key from env (do NOT hardcode it in the file)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit(
        "OPENROUTER_API_KEY is not set. "
        "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
    )
# -----------------------------


def main():
    # 1. Read the schema from disk
    with open("schema.sql", encoding="utf-8") as f:
        schema = f.read().strip()

    # 2. The one question we're testing
    question = "Who scored the most runs in 2024?"

    # 3. Build the prompt (schema + question)
    system_msg = (
        "You are a text-to-SQL generator. "
        "Given a database schema and a question, return a single SQL query that answers it. "
        "Use SQLite syntax. Return only the SQL query."
    )
    user_msg = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

    # 4. Connect via ChatOpenRouter
    #    - api_key is passed explicitly so you get a clear error if it's missing
    #    - temperature=0 -> deterministic, best for SQL generation
    #    - max_tokens=300 -> plenty for one SQL query
    llm = ChatOpenRouter(
        model=MODEL,
        api_key=api_key,
        temperature=0,
        max_tokens=300,
    )

    # 5. Send it and print what comes back
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)

    raw_sql = response.content

    print("=" * 60)
    print("MODEL:", MODEL)
    print("QUESTION:", question)
    print("=" * 60)
    print("RAW MODEL OUTPUT:")
    print(raw_sql)
    print("=" * 60)


if __name__ == "__main__":
    main()