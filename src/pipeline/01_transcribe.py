"""
Task 1: Auto Loader → Groq Whisper Large V3 → silver_transcripts Delta table.

One file per micro-batch (maxFilesPerTrigger=1) so that when Groq returns 429:
  - all previously transcribed files are already checkpointed
  - the rate-limited file is NOT checkpointed and will be retried on the next run
"""

import inspect
import os
import sys

from databricks.sdk.runtime import dbutils

sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import config  # noqa: E402
from logger import get_logger  # noqa: E402
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    element_at,
    regexp_extract,
    udf,
)
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder.getOrCreate()
log = get_logger("transcribe")

log.info(
    "Starting transcription task — catalog=%s schema=%s volume=%s",
    config.CATALOG,
    config.SCHEMA,
    config.VOLUME_NAME,
)
log.info("Source volume: %s", config.VOLUME_PATH)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{config.CATALOG}`.`{config.SCHEMA}`")

if not spark.catalog.tableExists(config.SILVER_TABLE):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {config.SILVER_TABLE} (
            filename        STRING NOT NULL,
            series          STRING,
            episode         STRING,
            transcript_text STRING,
            processed_at    TIMESTAMP
        )
        USING DELTA
    """)

log.info(
    "Fetching Groq API key from secret scope='%s' key='%s'",
    config.GROQ_SECRET_SCOPE,
    config.GROQ_SECRET_KEY,
)

try:
    _groq_key = dbutils.secrets.get(
        scope=config.GROQ_SECRET_SCOPE, key=config.GROQ_SECRET_KEY
    )
    log.info("Groq API key loaded successfully")
except Exception as e:
    raise RuntimeError(
        f"Could not fetch Groq API key from secret '{config.GROQ_SECRET_SCOPE}/{config.GROQ_SECRET_KEY}': {e}"
    )

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Progressive back-off delays (seconds) on consecutive 429 responses.
# After all retries are exhausted the UDF raises, the batch fails, and Auto
# Loader will retry the same file on the next trigger run.
_429_BACKOFFS = [10, 30, 120, 300]


@udf(returnType=StringType())
def transcribe_udf(content, path):
    import os
    import time

    import requests

    filename = os.path.basename(path)

    for attempt in range(len(_429_BACKOFFS) + 1):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {_groq_key}"},
            files={
                "file": (filename, bytes(content), "audio/mpeg"),
                "model": (None, "whisper-large-v3"),
                "language": (None, "hu"),
                "response_format": (None, "text"),
            },
            timeout=120,
        )

        if resp.status_code == 429:
            if attempt < len(_429_BACKOFFS):
                wait = _429_BACKOFFS[attempt]
                time.sleep(wait)
                continue
            # All retries exhausted — raise so the batch fails and the file
            # stays uncheckpointed for the next run.
            raise RuntimeError(
                f"Groq rate limit (429) on '{filename}' after {attempt + 1} attempts — will retry on next run"
            )

        if not resp.ok:
            raise RuntimeError(
                f"Groq {resp.status_code} on '{filename}': {resp.text[:300]}"
            )

        return resp.text.strip()


@udf(returnType=ArrayType(StringType()))
def parse_filename_udf(path):
    import os
    import re

    patterns = [
        re.compile(r"S(?P<series>\d+)E(?P<episode>\d+)", re.IGNORECASE),
        re.compile(r"(?<!\d)(?P<series>\d+)_(?P<episode>\d+)(?!\d)"),
    ]
    stem = os.path.splitext(os.path.basename(path))[0]
    for pat in patterns:
        m = pat.search(stem)
        if m:
            return [str(int(m.group("series"))), str(int(m.group("episode")))]
    return ["unknown", "unknown"]


stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", config.TRANSCRIBE_CHECKPOINT + "/schema")
    .option("cloudFiles.maxFilesPerTrigger", config.WHISPER_MAX_FILES_PER_TRIGGER)
    .option("pathGlobFilter", "*.mp3")
    .load(config.VOLUME_PATH)
)

parsed = stream.withColumn("_se", parse_filename_udf(col("path")))

transcribed = (
    parsed.withColumn("filename", regexp_extract(col("path"), r"[^/]+$", 0))
    .withColumn("series", element_at(col("_se"), 1))
    .withColumn("episode", element_at(col("_se"), 2))
    .withColumn("transcript_text", transcribe_udf(col("content"), col("path")))
    .withColumn("processed_at", current_timestamp())
    .select("filename", "series", "episode", "transcript_text", "processed_at")
)

(
    transcribed.writeStream.option(
        "checkpointLocation", config.TRANSCRIBE_CHECKPOINT + "/data"
    )
    .trigger(availableNow=True)
    .toTable(config.SILVER_TABLE)
    .awaitTermination()
)

log.info("Transcription task finished.")
