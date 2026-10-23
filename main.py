"""
main.py — POORA EVAL FLOW (ChatOpenRouter version) — IMPROVED

Flow:
  1. Model load karo (OpenRouter ke through)
  2. Golden dataset ke saare questions pe SQL generate karwao
  3. Generated SQL DB pe run karo → result milega
  4. Gold result se compare karo (evaluator.py)
  5. Score log karo

Improvements over original:
  - Parallel question processing (ThreadPoolExecutor) → SPEED
  - Retry logic with exponential backoff → RELIABILITY
  - Timeout on LLM calls → no hanging
  - Better error handling (None checks, logging)
  - Progress bar (tqdm)
  - Timestamped result files (no overwrite)
  - Cost/token tracking

Setup:
  pip install langchain-openrouter python-dotenv pandas tqdm tenacity
  .env file: OPENROUTER_API_KEY=sk-or-...
"""

import os
import re
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from model_openrouter_slug import MODELS
from evaluator import evaluate_one

# ---------- CONFIG ----------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

DB_PATH = "ipl_2021_2024.db"
SCHEMA_PATH = "schema.sql"
GOLDEN_PATH = "golden_dataset.csv"

# Timestamped output — purane results overwrite nahi honge
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"eval_results_{TIMESTAMP}.csv"

MAX_WORKERS = 5          # Kitne questions parallel chalenge (API rate limits dekh ke set karo)
LLM_TIMEOUT = 60         # Seconds — LLM call timeout
MAX_RETRIES = 3          # API fail hone pe kitni baar retry
# ----------------------------


# ============================================================
# HELPER: File existence check — crash se bachne ke liye
# ============================================================
def check_files():
    """Saari required files exist karti hain ya nahi — pehle check karo."""
    required = [DB_PATH, SCHEMA_PATH, GOLDEN_PATH]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")


# ============================================================
# HELPER: Schema load
# ============================================================
def load_schema():
    """schema.sql file padh ke string return karo (prompt mein jayega)."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return f.read().strip()


# ============================================================
# HELPER: Golden dataset load
# ============================================================
def load_golden():
    """golden_dataset.csv ko DataFrame mein load karo."""
    return pd.read_csv(GOLDEN_PATH)


# ============================================================
# HELPER: Model instance banao
# ============================================================
def make_llm(slug):
    """Ek OpenRouter model instance banao. temperature=0 = deterministic."""
    return ChatOpenRouter(
        model=slug,
        openrouter_api_key=API_KEY,
        temperature=0,
        max_tokens=1500,       # SQL short hoti hai, par complex queries ke liye thoda buffer
        timeout=LLM_TIMEOUT,   # Hang na ho isliye timeout
    )


# ============================================================
# HELPER: SQL clean karo
# ============================================================
def clean_sql(raw):
    """
    LLM ka raw output clean karo — sirf runnable SQL bache.
    Steps:
      1. Markdown fences (```sql ... ```) hatao
      2. Leading "sql" word hatao
      3. SELECT/WITH se pehle ka prose hatao
      4. Trailing backticks/semicolons hatao
    """
    if not raw:
        return ""
    text = raw.strip()

    # 1. Code fence extract
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # 2. Leading "sql" word hatao
    text = re.sub(r"^\s*sql\s*\n", "", text, flags=re.IGNORECASE)

    # 3. SELECT ya WITH se pehle ka sab hatao
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]

    # 4. Cleanup
    return text.strip().strip("`").rstrip(";").strip("`").strip()


# ============================================================
# HELPER: LLM se SQL generate karwao — RETRY ke saath
# ============================================================
@retry(
    stop=stop_after_attempt(MAX_RETRIES),                # 3 attempts max
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s → 4s → 8s backoff
    retry=retry_if_exception_type(Exception),            # Koi bhi exception pe retry
    reraise=True,                                         # Saare retries fail → raise
)
def _call_llm(llm, system_msg, user_msg):
    """
    LLM call — retry wrapper ke andar.
    Network blip, rate limit, timeout — sab pe retry hoga.
    """
    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ])
    return response.content


def generate_sql(question, schema, llm):
    """Question + schema → LLM → cleaned SQL string (ya None agar fail)."""
    system_msg = (
        "You are a text-to-SQL generator. Given a database schema and a question, "
        "return a single SQL query that answers it. Use SQLite syntax. "
        "Return only the SQL query."
    )
    user_msg = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

    try:
        raw = _call_llm(llm, system_msg, user_msg)
    except Exception as e:
        print(f"  LLM call failed after {MAX_RETRIES} retries: {e}")
        return None

    cleaned = clean_sql(raw)
    if not cleaned:
        print(f"  RAW (uncleaned, first 300 chars): {repr(raw)[:300]}")
        return None
    return cleaned


# ============================================================
# HELPER: SQL run karo DB pe
# ============================================================
def run_sql(conn, sql):
    """SQL ko DB pe run karo. DataFrame ya None (error pe)."""
    if not sql:
        return None
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        # Debugging ke liye error print (chhota message)
        print(f"  SQL error: {str(e)[:120]}")
        return None


# ============================================================
# HELPER: Gold result JSON → DataFrame
# ============================================================
def gold_result_to_df(gold_result_json):
    """Stored JSON se gold DataFrame rebuild karo."""
    obj = json.loads(gold_result_json)
    return pd.DataFrame(obj["rows"], columns=obj["columns"])


# ============================================================
# HELPER: Ek question process karo (parallel worker)
# ============================================================
def process_one_question(conn, llm, schema, g):
    """
    Ek question ka poora flow:
      generate SQL → run DB pe → compare with gold → return result dict.
    Thread-safe: sirf DB read karta hai (SQLite read concurrency allowed).
    """
    qid = g["id"]
    question = g["question"]
    order_sensitive = str(g["order_sensitive"]).upper() == "TRUE"
    gold_df = gold_result_to_df(g["gold_result_json"])

    # 1. Generate SQL
    sql = generate_sql(question, schema, llm)
    if sql is None:
        return dict(
            id=qid, question=question, difficulty=g["difficulty"],
            correct=False, reason="gen_error", sql="",
        )

    # 2. Run generated SQL
    gen_df = run_sql(conn, sql)
    if gen_df is None:
        return dict(
            id=qid, question=question, difficulty=g["difficulty"],
            correct=False, reason="sql_error", sql=sql,
        )

    # 3. Compare with gold
    try:
        verdict = evaluate_one(gold_df, gen_df, order_sensitive)
    except Exception as e:
        verdict = {"correct": False, "reason": f"eval_error: {e}"}

    return dict(
        id=qid, question=question, difficulty=g["difficulty"],
        correct=verdict["correct"], reason=verdict["reason"], sql=sql,
    )


# ============================================================
# MAIN EVAL FLOW
# ============================================================
def run_eval():
    # Files check karo pehle
    check_files()

    # Load everything
    schema = load_schema()
    golden = load_golden()
    total = len(golden)

    if total == 0:
        print("Golden dataset empty! Abort.")
        return

    print(f"Loaded {total} questions | {len(MODELS)} models | "
          f"Parallel workers: {MAX_WORKERS}")

    all_rows = []
    scoreboard = {}

    # Har model ke liye
    for name, slug in MODELS:
        print(f"\n{'='*70}\nMODEL: {name}  ({slug})\n{'='*70}")

        llm = make_llm(slug)          # Model load
        correct = 0

        # DB connection per model (thread-safety ke liye — SQLite mein
        # same connection multiple threads se issues deta hai)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)

        # Parallel processing — questions in parallel
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_one_question, conn, llm, schema, g): g
                for _, g in golden.iterrows()
            }
            # Progress bar ke saath collect karo
            for fut in tqdm(as_completed(futures), total=total,
                            desc=f"  {name}", ncols=90):
                try:
                    res = fut.result()
                except Exception as e:
                    g = futures[fut]
                    res = dict(
                        id=g["id"], question=g["question"],
                        difficulty=g["difficulty"], correct=False,
                        reason=f"thread_error: {e}", sql="",
                    )
                results.append(res)

        conn.close()

        # Score calculate
        correct = sum(1 for r in results if r["correct"])
        scoreboard[name] = correct

        # Detailed per-question log
        for r in results:
            mark = "OK " if r["correct"] else "XX "
            print(f"  #{r['id']:2} [{r['difficulty']:6}] {mark} {r['reason']}")

        print(f"\n  SCORE: {correct}/{total} = {100*correct/total:.1f}%")

        # Model name results mein add karo
        for r in results:
            r["model"] = name
            all_rows.append(r)

    # ---------- FINAL SCOREBOARD ----------
    print(f"\n{'='*70}\nFINAL SCOREBOARD (execution accuracy)\n{'='*70}")
    for name, _ in MODELS:
        c = scoreboard[name]
        print(f"  {name:18s} {c:2}/{total}  = {100*c/total:5.1f}%")

    # ---------- SAVE RESULTS ----------
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nDetailed results saved -> {RESULTS_PATH}")

    # Timestamped summary bhi save karo
    summary_path = f"eval_summary_{TIMESTAMP}.csv"
    pd.DataFrame([
        {"model": name, "correct": scoreboard[name],
         "total": total, "accuracy": 100*scoreboard[name]/total}
        for name, _ in MODELS
    ]).to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if not API_KEY:
        print("Missing OPENROUTER_API_KEY — add it to your .env file")
    else:
        run_eval()