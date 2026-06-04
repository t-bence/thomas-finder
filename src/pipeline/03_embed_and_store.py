"""
Task 3: silver_transcripts_clean → gold_episodes Delta table.

Streams new rows from silver_transcripts_clean (checkpoint-based idempotency)
and writes one row per episode to gold_episodes. No embedding is computed here —
Vector Search uses managed embeddings via embedding_source_column.

Change Data Feed is required on gold_episodes for the Delta Sync VS index.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import config  # noqa: E402
from logger import get_logger  # noqa: E402

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
log = get_logger("embed_and_store")

log.info(
    "Starting embed-and-store task — catalog=%s schema=%s",
    config.CATALOG,
    config.SCHEMA,
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{config.CATALOG}`.`{config.SCHEMA}`")

if not spark.catalog.tableExists(config.GOLD_TABLE):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {config.GOLD_TABLE} (
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
        f"ALTER TABLE {config.GOLD_TABLE} "
        "SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )

stream = spark.readStream.table(config.CLEAN_TABLE)

(
    stream
    .select("filename", "series", "episode", "transcript_text", "processed_at")
    .writeStream
    .option("checkpointLocation", config.INDEX_CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(config.GOLD_TABLE)
    .awaitTermination()
)

log.info("Embed-and-store task finished.")
