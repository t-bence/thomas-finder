import os
import re
import sys

# Parameters passed from the job task (positional args: catalog schema [volume_name])
_args = sys.argv[1:]
CATALOG = _args[0] if len(_args) > 0 else os.getenv("CATALOG", "workspace")
SCHEMA = _args[1] if len(_args) > 1 else os.getenv("SCHEMA", "thomas")
VOLUME_NAME = _args[2] if len(_args) > 2 else os.getenv("VOLUME_NAME", "episodes")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

_CHECKPOINTS = f"{VOLUME_PATH}/_checkpoints"
TRANSCRIBE_CHECKPOINT = f"{_CHECKPOINTS}/transcribe"
CLEAN_CHECKPOINT = f"{_CHECKPOINTS}/clean"
INDEX_CHECKPOINT = f"{_CHECKPOINTS}/index"

SILVER_TABLE = f"`{CATALOG}`.`{SCHEMA}`.silver_transcripts"
CLEAN_TABLE = f"`{CATALOG}`.`{SCHEMA}`.silver_transcripts_clean"
GOLD_TABLE = f"`{CATALOG}`.`{SCHEMA}`.gold_episodes"

VS_ENDPOINT = "thomas-finder-vs"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.episode_chunks_index"

EMBED_MODEL = "databricks-qwen3-embedding-0-6b"

CLEAN_MODEL = "databricks-meta-llama-3-3-70b-instruct"

GROQ_SECRET_SCOPE = "Thomas"
GROQ_SECRET_KEY = "Groq_key"
# Process one file per micro-batch so a 429 stops cleanly without losing progress
WHISPER_MAX_FILES_PER_TRIGGER = 1

# Two filename patterns, tried in order:
#   1. "Thomas, a gozmozdony S16E17 720o (1).mp3"  → SxxExx
#   2. "16_02.mp3"                                   → series_episode
_PATTERNS = [
    re.compile(r"S(?P<series>\d+)E(?P<episode>\d+)", re.IGNORECASE),
    re.compile(r"(?<!\d)(?P<series>\d+)_(?P<episode>\d+)(?!\d)"),
]


def parse_filename(path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    for pat in _PATTERNS:
        m = pat.search(stem)
        if m:
            series = str(int(m.group("series")))
            episode = str(int(m.group("episode")))
            return series, episode
    return "unknown", "unknown"
