# Thomas Kereső

Find Thomas the Tank Engine episodes (Hungarian audio) by describing what you remember from the story.

## How it works

1. **Transcribe** — Auto Loader detects new `.mp3` files in a Databricks Volume and sends each one to a Whisper Large V3 Model Serving endpoint. Results land in a `silver_transcripts` Delta table. The Auto Loader checkpoint ensures every file is processed exactly once, even across reruns.
2. **Embed** — Transcripts are chunked and embedded with `databricks-qwen3-embedding-0-6b`. Embeddings are written to a `gold_episode_chunks` Delta table with Change Data Feed enabled.
3. **Index** — A Databricks Vector Search Delta Sync index is created (or synced) from the gold table.
4. **Search** — A Streamlit app embeds the user's query with the same Qwen3 model and returns the top matching episodes by semantic similarity. No LLM needed.

## Prerequisites

- Databricks workspace (Free tier works — see note below)
- Databricks CLI configured: `databricks configure`
- A Unity Catalog Volume containing `.mp3` files named in one of these formats:
  - `Thomas, a gozmozdony S16E17 720p.mp3` (SxxExx)
  - `16_02.mp3` (series_episode)

## Deployment

```bash
databricks bundle validate
databricks bundle deploy          # deploys Whisper endpoint + job + app
databricks bundle run thomas_finder_job   # transcribe → embed → index
databricks bundle run thomas_finder_app  # start the Streamlit app
```

The Whisper endpoint needs a minute or two to come online before the job runs.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `main` | Unity Catalog catalog |
| `schema` | `thomas` / `thomas_dev` | Schema (per target) |
| `volume_name` | `episodes` | Volume containing the `.mp3` files |

Override with `--var` at deploy time: `databricks bundle deploy --var catalog=my_catalog`

Update `src/app/app.yaml` env vars if you change catalog/schema.

## Targets

| Target | Schema | Notes |
|--------|--------|-------|
| `dev` (default) | `thomas_dev` | Safe to redeploy and destroy |
| `prod` | `thomas` | |

## Project layout

```
databricks.yml          # bundle root + targets
resources/
  job.yml               # processing job (3 tasks, serverless)
  app.yml               # Databricks App
  whisper_endpoint.yml  # Whisper Large V3 Model Serving endpoint
src/
  pipeline/
    config.py           # shared config + filename parsing
    01_transcribe.py    # Auto Loader + Whisper endpoint → silver table
    02_embed.py         # chunk + Qwen3 embeddings → gold table
    03_setup_vector_search.py
  app/
    app.py              # Streamlit UI
    app.yaml            # app runtime config
    requirements.txt
```

## Note: CPU compute on Databricks Free tier

The Whisper endpoint (`system.ai.whisper_large_v3`) is configured with `workload_type: CPU` because Databricks Free tier does not support GPU Model Serving endpoints. Transcription will be noticeably slower than on GPU — expect roughly 5–10× longer per episode. For a one-time batch over a few hundred episodes this is acceptable; upgrade to a paid tier and switch `workload_type` to `GPU_SMALL` in `resources/whisper_endpoint.yml` for faster processing.
