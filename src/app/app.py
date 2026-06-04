import os

import streamlit as st

CATALOG = os.getenv("CATALOG", "main")
SCHEMA = os.getenv("SCHEMA", "thomas")
VS_ENDPOINT = os.getenv("VS_ENDPOINT", "thomas-finder-vs")
VS_INDEX = os.getenv("VS_INDEX", f"{CATALOG}.{SCHEMA}.episode_chunks_index")
EMBED_MODEL = "databricks-qwen3-embedding-0-6b"
NUM_RESULTS = 5


@st.cache_resource
def get_vs_index():
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    return vsc.get_index(VS_ENDPOINT, VS_INDEX)


@st.cache_resource
def get_embed_client():
    from mlflow.deployments import get_deploy_client

    return get_deploy_client("databricks")


def embed_query(text: str) -> list[float]:
    response = get_embed_client().predict(
        endpoint=EMBED_MODEL,
        inputs={"input": [text]},
    )
    return response["data"][0]["embedding"]


def search(query: str) -> list[dict]:
    embedding = embed_query(query)
    raw = get_vs_index().similarity_search(
        query_vector=embedding,
        columns=["filename", "series", "episode", "chunk_text"],
        num_results=NUM_RESULTS * 3,  # fetch extra to deduplicate per episode
    )
    columns = [c["name"] for c in raw.get("result", {}).get("columns", [])]
    rows = raw.get("result", {}).get("data_array", [])

    seen: set[tuple] = set()
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        key = (item["series"], item["episode"])
        if key not in seen:
            seen.add(key)
            results.append(item)
        if len(results) >= NUM_RESULTS:
            break
    return results


st.set_page_config(page_title="Thomas Kereső", page_icon="🚂", layout="centered")
st.title("🚂 Thomas Kereső")
st.caption("Írd le, amire emlékszel — megtaláljuk, melyik részből való!")

query = st.text_area(
    label="Mire emlékszel?",
    placeholder="Pl.: Thomas kap egy piros vontatót és versenyez Gordonnal a hegyen...",
    height=100,
)

if st.button("🔍 Keresés", use_container_width=True, type="primary") and query.strip():
    with st.spinner("Keresés..."):
        try:
            hits = search(query.strip())
        except Exception as e:
            st.error(f"Hiba a keresés során: {e}")
            hits = []

    if not hits:
        st.info("Nincs találat. Próbálj más szavakkal!")
    else:
        st.subheader(f"Top {len(hits)} találat")
        for hit in hits:
            series = hit.get("series", "?")
            episode = hit.get("episode", "?")
            filename = hit.get("filename", "")
            snippet = hit.get("chunk_text", "")[:450]
            if len(hit.get("chunk_text", "")) > 450:
                snippet += "…"

            with st.container(border=True):
                st.markdown(f"### {series}. sorozat, {episode}. rész")
                if filename:
                    st.caption(f"`{filename}`")
                st.write(snippet)
