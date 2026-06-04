"""
Task 2: silver_transcripts → Llama cleaning → silver_transcripts_clean.

Streams new rows from silver_transcripts (Delta readStream, checkpoint-based
idempotency) and uses ai_query to remove the opening broadcast disclaimer and
closing producer credits from each transcript.
"""

import inspect
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)
import config  # noqa: E402
from logger import get_logger  # noqa: E402
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, expr, lit

spark = SparkSession.builder.getOrCreate()
log = get_logger("clean")

log.info(
    "Starting clean task — catalog=%s schema=%s model=%s",
    config.CATALOG,
    config.SCHEMA,
    config.CLEAN_MODEL,
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{config.CATALOG}`.`{config.SCHEMA}`")

if not spark.catalog.tableExists(config.CLEAN_TABLE):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {config.CLEAN_TABLE} (
            filename        STRING NOT NULL,
            series          STRING,
            episode         STRING,
            transcript_text STRING,
            processed_at    TIMESTAMP
        )
        USING DELTA
    """)

_PROMPT_PREFIX = (
    "You are cleaning a Hungarian Thomas the Tank Engine episode transcript. "
    "Remove the opening broadcast disclaimer — it typically starts with phrases like "
    "'A következő műsor' or similar Hungarian broadcast text"
    "before the actual story begins. "
    "Also remove things like or 'A TORONTOS' -- this is probably "
    "Remove any closing producer credits, copyright notices, or technical information "
    "after the story ends. "
    "Make sure you keep the names in English, such as Thomas, Emily, Gordon, etc. "
    "Do not change the names to Hungarian."
    "Clean the text, remove missing letters, interpolate broken words."
    "Return only the story content, nothing else. Do not add any commentary.\n\n"
    "Transcript:\n"
)

stream = spark.readStream.table(config.SILVER_TABLE)

cleaned = (
    stream.withColumn("_prompt", concat(lit(_PROMPT_PREFIX), col("transcript_text")))
    .withColumn(
        "transcript_text",
        expr(f"ai_query('{config.CLEAN_MODEL}', _prompt)"),
    )
    .drop("_prompt")
    .select("filename", "series", "episode", "transcript_text", "processed_at")
)

(
    cleaned.writeStream.option("checkpointLocation", config.CLEAN_CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(config.CLEAN_TABLE)
    .awaitTermination()
)

log.info("Clean task finished.")
