"""
Task 4: Create Vector Search endpoint + managed-embedding Delta Sync index (idempotent).

On first run: creates both the endpoint and the index.
On subsequent runs: triggers a sync to pick up newly added episodes.

Embeddings are computed by Databricks Vector Search using the configured
embedding model — no embedding column is stored in gold_episodes.
"""

import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import config  # noqa: E402
from databricks.vector_search.client import VectorSearchClient
from logger import get_logger  # noqa: E402

log = get_logger("vector_search")
vsc = VectorSearchClient(disable_notice=True)

log.info(
    "Starting vector search setup — endpoint=%s index=%s embed_model=%s",
    config.VS_ENDPOINT,
    config.VS_INDEX_NAME,
    config.EMBED_MODEL,
)


def _endpoint_exists() -> bool:
    result = vsc.list_endpoints()
    return any(e["name"] == config.VS_ENDPOINT for e in result.get("endpoints", []))


def _wait_for_endpoint():
    log.info("Waiting for endpoint '%s' to come online...", config.VS_ENDPOINT)
    while True:
        status = vsc.get_endpoint(config.VS_ENDPOINT)
        state = status.get("endpoint_status", {}).get("state", "")
        log.info("  Endpoint state: %s", state)
        if state == "ONLINE":
            break
        if state in ("PROVISIONING_FAILED", "OFFLINE"):
            raise RuntimeError(f"Endpoint entered state: {state}")
        time.sleep(15)
    log.info("Endpoint is online.")


if not _endpoint_exists():
    log.info("Creating Vector Search endpoint: %s", config.VS_ENDPOINT)
    vsc.create_endpoint(name=config.VS_ENDPOINT, endpoint_type="STANDARD")
    _wait_for_endpoint()
else:
    log.info("Endpoint '%s' already exists.", config.VS_ENDPOINT)

try:
    index = vsc.get_index(config.VS_ENDPOINT, config.VS_INDEX_NAME)
    log.info("Index '%s' already exists — triggering sync...", config.VS_INDEX_NAME)
    index.sync()
    log.info("Sync triggered. The index will update in the background.")
except Exception:
    log.info("Index not found — creating managed-embedding Delta Sync index: %s", config.VS_INDEX_NAME)
    vsc.create_delta_sync_index(
        endpoint_name=config.VS_ENDPOINT,
        source_table_name=config.GOLD_TABLE.replace("`", ""),
        index_name=config.VS_INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key="filename",
        embedding_source_column="transcript_text",
        embedding_model_endpoint_name=config.EMBED_MODEL,
    )
    log.info("Index created — initial sync running in the background.")

log.info("Vector search setup complete.")
