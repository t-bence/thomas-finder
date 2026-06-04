"""
Task 3: Create Vector Search endpoint + managed-embedding Delta Sync index (idempotent).

On first run: creates both the endpoint and the index.
On subsequent runs: triggers a sync to pick up newly added episodes.

Embeddings are computed by Databricks Vector Search using the configured
embedding model — no embedding column is stored in silver_transcripts_clean.
"""

import logging
import sys
import time

from databricks.vector_search.client import VectorSearchClient

# --- config ---
_args = sys.argv[1:]
CLEAN_TABLE = _args[0]
VS_ENDPOINT = _args[1]
VS_INDEX_NAME = _args[2]
EMBED_MODEL = _args[3]

# --- logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("vector_search")

# --- main ---
vsc = VectorSearchClient(disable_notice=True)

log.info(
    "Starting vector search setup — endpoint=%s index=%s embed_model=%s",
    VS_ENDPOINT, VS_INDEX_NAME, EMBED_MODEL,
)


def _endpoint_exists() -> bool:
    result = vsc.list_endpoints()
    return any(e["name"] == VS_ENDPOINT for e in result.get("endpoints", []))


def _wait_for_endpoint():
    log.info("Waiting for endpoint '%s' to come online...", VS_ENDPOINT)
    while True:
        state = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state", "")
        log.info("  Endpoint state: %s", state)
        if state == "ONLINE":
            break
        if state in ("PROVISIONING_FAILED", "OFFLINE"):
            raise RuntimeError(f"Endpoint entered state: {state}")
        time.sleep(15)
    log.info("Endpoint is online.")


if not _endpoint_exists():
    log.info("Creating Vector Search endpoint: %s", VS_ENDPOINT)
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    _wait_for_endpoint()
else:
    log.info("Endpoint '%s' already exists.", VS_ENDPOINT)

try:
    index = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    log.info("Index '%s' already exists — triggering sync...", VS_INDEX_NAME)
    index.sync()
    log.info("Sync triggered. The index will update in the background.")
except Exception:
    log.info("Index not found — creating managed-embedding Delta Sync index: %s", VS_INDEX_NAME)
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=CLEAN_TABLE.replace("`", ""),
        index_name=VS_INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key="filename",
        embedding_source_column="transcript_text",
        embedding_model_endpoint_name=EMBED_MODEL,
    )
    log.info("Index created — initial sync running in the background.")

log.info("Vector search setup complete.")
