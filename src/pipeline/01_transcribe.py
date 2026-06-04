"""
Task 1: Auto Loader → Groq Whisper Large V3 → silver_transcripts Delta table.

One file per micro-batch (maxFilesPerTrigger=1) so that when Groq returns 429:
  - all previously transcribed files are already checkpointed
  - the rate-limited file is NOT checkpointed and will be retried on the next run
"""

import logging
import sys

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, element_at, regexp_extract, udf
from pyspark.sql.types import ArrayType, StringType

# --- config ---
_args = sys.argv[1:]
CATALOG_SCHEMA = _args[0]   # e.g. "workspace.thomas_dev"
VOLUME_PATH = _args[1]
SILVER_TABLE = _args[2]
TRANSCRIBE_CHECKPOINT = f"{VOLUME_PATH}/_checkpoints/transcribe"

GROQ_SECRET_SCOPE = "Thomas"
GROQ_SECRET_KEY = "Groq_key"
WHISPER_MAX_FILES_PER_TRIGGER = 1
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# Progressive back-off delays (seconds) on consecutive 429 responses.
# After all retries are exhausted the UDF raises, the batch fails, and Auto
# Loader will retry the same file on the next trigger run.
_429_BACKOFFS = [10, 30, 120, 300]

# --- logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("transcribe")

# --- main ---
spark = SparkSession.builder.getOrCreate()

log.info("Starting transcription task — volume=%s silver=%s", VOLUME_PATH, SILVER_TABLE)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_SCHEMA}")

if not spark.catalog.tableExists(SILVER_TABLE):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
            filename        STRING NOT NULL,
            series          STRING,
            episode         STRING,
            transcript_text STRING,
            processed_at    TIMESTAMP
        )
        USING DELTA
    """)

log.info("Fetching Groq API key from secret scope='%s' key='%s'", GROQ_SECRET_SCOPE, GROQ_SECRET_KEY)
try:
    _groq_key = dbutils.secrets.get(scope=GROQ_SECRET_SCOPE, key=GROQ_SECRET_KEY)
    log.info("Groq API key loaded successfully")
except Exception as e:
    raise RuntimeError(
        f"Could not fetch Groq API key from secret '{GROQ_SECRET_SCOPE}/{GROQ_SECRET_KEY}': {e}"
    )


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
                time.sleep(_429_BACKOFFS[attempt])
                continue
            raise RuntimeError(
                f"Groq rate limit (429) on '{filename}' after {attempt + 1} attempts — will retry on next run"
            )

        if not resp.ok:
            raise RuntimeError(f"Groq {resp.status_code} on '{filename}': {resp.text[:300]}")

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
    .option("cloudFiles.schemaLocation", TRANSCRIBE_CHECKPOINT + "/schema")
    .option("cloudFiles.maxFilesPerTrigger", WHISPER_MAX_FILES_PER_TRIGGER)
    .option("pathGlobFilter", "*.mp3")
    .load(VOLUME_PATH)
)

parsed = stream.withColumn("_se", parse_filename_udf(col("path")))

transcribed = (
    parsed
    .withColumn("filename", regexp_extract(col("path"), r"[^/]+$", 0))
    .withColumn("series", element_at(col("_se"), 1))
    .withColumn("episode", element_at(col("_se"), 2))
    .withColumn("transcript_text", transcribe_udf(col("content"), col("path")))
    .withColumn("processed_at", current_timestamp())
    .select("filename", "series", "episode", "transcript_text", "processed_at")
)

(
    transcribed.writeStream
    .option("checkpointLocation", TRANSCRIBE_CHECKPOINT + "/data")
    .trigger(availableNow=True)
    .toTable(SILVER_TABLE)
    .awaitTermination()
)

log.info("Transcription task finished.")
