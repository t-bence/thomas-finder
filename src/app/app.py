import os

import streamlit as st
from databricks.sdk import WorkspaceClient

VS_INDEX = os.getenv("VECTOR_SEARCH_INDEX")
if not VS_INDEX:
    raise ValueError("Missing vector search index")

NUM_RESULTS = 5

w = WorkspaceClient()


def search(query: str) -> list[dict]:
    results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        query_text=query,
        columns=["filename", "series", "episode", "transcript_text"],
        num_results=NUM_RESULTS,
    )
    return [
        {
            "filename": item[0],
            "series": item[1],
            "episode": item[2],
            "transcript_text": item[3],
        }
        for item in results.result.data_array
    ]


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
            snippet = hit.get("transcript_text", "")[:450]
            if len(hit.get("transcript_text", "")) > 450:
                snippet += "…"

            with st.container(border=True):
                st.markdown(f"### {series}. sorozat, {episode}. rész")
                if filename:
                    st.caption(f"`{filename}`")
                st.write(snippet)
