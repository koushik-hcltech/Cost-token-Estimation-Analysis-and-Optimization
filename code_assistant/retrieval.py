"""Retrieval backends for Code Assistant Agent.

For local validation both corpora live in in-memory ChromaDB. In Qwiklabs the
DocsAgent corpus is swapped for a managed Agent Search datastore; its tool
records cost as a per-query ``vais_query`` charge, so the cost model is the same
whether the backend is managed Search or the local stand-in.
"""
from google import genai
from google.genai import types as genai_types

_clients: dict = {}


def genai_client(project: str, location: str):
    key = (project, location)
    if key not in _clients:
        # Opt into the SDK's transient-error retry policy (defaults: 5 attempts,
        # exponential backoff, 408/429/5xx); without this the client tries once.
        _clients[key] = genai.Client(
            vertexai=True, project=project, location=location,
            http_options=genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions()))
    return _clients[key]


def embed_texts(texts, embed_model, project, location,
                ledger=None, query_id="_", agent="embedding"):
    """Embed texts via Vertex; record billable input tokens in the ledger (duck-typed)."""
    client = genai_client(project, location)
    resp = client.models.embed_content(model=embed_model, contents=texts)
    if not getattr(resp, "embeddings", None):
        # Fail loudly rather than return a partial result the caller would misread.
        raise RuntimeError(f"embedding call returned no embeddings for {len(texts)} text(s)")
    if ledger is not None:
        tokens = sum(int(getattr(e.statistics, "token_count", 0) or 0)
                     for e in resp.embeddings if getattr(e, "statistics", None) is not None)
        if not tokens:
            tokens = max(1, sum(len(t) for t in texts) // 4)   # rough fallback if unreported
        ledger.add(query_id=query_id, agent=agent, call_type="embedding",
                   model="embedding", embed_tokens=tokens)
    return [e.values for e in resp.embeddings]


def build_index(rows, embed_model, project, location, *, name="corpus", ledger=None):
    """Build an in-memory Chroma collection from rows ({'id','text'})."""
    import chromadb
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name)
    docs = [r["text"] for r in rows]
    ids = [str(r["id"]) for r in rows]
    embs = embed_texts(docs, embed_model, project, location, ledger,
                       query_id="__index__", agent="index")
    col.add(ids=ids, documents=docs, embeddings=embs)
    return col


def make_lookup_tool(collection, embed_model, project, location, ledger, qid_holder,
                     *, agent, top_k=5, record_as="embedding"):
    """Build a (query:str)->str function for an ADK FunctionTool.

    record_as='vais'     -> bill a managed per-query charge (DocsAgent)
    record_as='embedding'-> bill the query embedding (CommunityAgent / local vector DB)
    """
    def lookup(query: str) -> str:
        bill = ledger if record_as == "embedding" else None
        q_emb = embed_texts([query], embed_model, project, location, bill,
                            qid_holder["v"], agent)[0]
        res = collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        if record_as == "vais":
            ledger.add(query_id=qid_holder["v"], agent=agent, call_type="retrieval",
                       model="vais_query", retrieval_queries=1)
        return "\n\n".join(docs) if docs else "No results found."

    lookup.__name__ = f"{agent.lower()}_lookup"
    lookup.__doc__ = f"Look up relevant {agent} context for a coding question or error message."
    return lookup
