"""
Task 2: silver_transcripts → Llama cleaning → silver_transcripts_clean.

Streams new rows from silver_transcripts (Delta readStream, checkpoint-based
idempotency) and uses ai_query to remove the opening broadcast disclaimer and
closing producer credits from each transcript.
"""

import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, expr, lit

# --- config ---
_args = sys.argv[1:]
CATALOG_SCHEMA = _args[0]  # e.g. "workspace.thomas_dev"
VOLUME_PATH = _args[1]
SILVER_TABLE = _args[2]
CLEAN_TABLE = _args[3]
CLEAN_MODEL = _args[4]
CLEAN_CHECKPOINT = f"{VOLUME_PATH}/_checkpoints/clean"

# --- logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("clean")

# --- main ---
spark = SparkSession.builder.getOrCreate()

log.info("Starting clean task — schema=%s model=%s", CATALOG_SCHEMA, CLEAN_MODEL)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_SCHEMA}")

if not spark.catalog.tableExists(CLEAN_TABLE):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CLEAN_TABLE} (
            filename        STRING NOT NULL,
            series          STRING,
            episode         STRING,
            transcript_text STRING,
            processed_at    TIMESTAMP
        )
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
else:
    spark.sql(
        f"ALTER TABLE {CLEAN_TABLE} "
        "SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )

_PROMPT_PREFIX = (
    "You are cleaning a Hungarian Thomas the Tank Engine episode transcript. "
    "Remove the opening broadcast disclaimer — it typically starts with phrases like "
    "'A következő műsor' or similar Hungarian broadcast text"
    "before the actual story begins. "
    "Also remove things like or 'A TORONTOS' -- this is probably English text that was"
    "misunderstood by the text to speech model."
    "Remove any closing producer credits, copyright notices, or technical information "
    "after the story ends. "
    "Make sure you keep the names in English, such as Thomas, Emily, Gordon, etc. "
    "Do not change the names to Hungarian."
    "Clean the text, remove missing letters, interpolate broken words."
    "Return only the story content, nothing else. Do not add any commentary.\n\n"
    "Transcript:\n"
)

stream = spark.readStream.table(SILVER_TABLE)

cleaned = (
    stream.withColumn("_prompt", concat(lit(_PROMPT_PREFIX), col("transcript_text")))
    .withColumn("transcript_text", expr(f"ai_query('{CLEAN_MODEL}', _prompt)"))
    .drop("_prompt")
    .select("filename", "series", "episode", "transcript_text", "processed_at")
)

(
    cleaned.writeStream.option("checkpointLocation", CLEAN_CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(CLEAN_TABLE)
    .awaitTermination()
)

log.info("Clean task finished.")
