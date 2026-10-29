"""Real managed retrieval via Agent Search (Discovery Engine).

Provisions a GENERIC structured data store + STANDARD search engine, imports the
docs corpus inline (JSON, no GCS), and exposes a lookup tool DocsAgent calls.
Retrieval cost is recorded as a managed 'vais_query' charge in the usage ledger.

This is the production replacement for the in-memory Chroma stand-in used for
DocsAgent during local validation.
"""
import json
import time
from google.cloud import discoveryengine_v1 as de
from google.api_core import retry as core_retry
from google.api_core.exceptions import AlreadyExists, GoogleAPICallError


def _collection(project, location):
    return f"projects/{project}/locations/{location}/collections/default_collection"


def _datastore(project, location, ds_id):
    return f"{_collection(project, location)}/dataStores/{ds_id}"


def serving_config(project, location, engine_id):
    return f"{_collection(project, location)}/engines/{engine_id}/servingConfigs/default_search"


def provision(project, location, ds_id, engine_id, rows):
    """Idempotently create the data store, import docs, and create the engine.
    Returns the serving-config path. (Indexing continues async after this.)"""
    parent = _collection(project, location)

    ds_client = de.DataStoreServiceClient()
    try:
        op = ds_client.create_data_store(
            parent=parent, data_store_id=ds_id,
            data_store=de.DataStore(
                display_name="Code Assistant Agent Docs",
                industry_vertical=de.IndustryVertical.GENERIC,
                solution_types=[de.SolutionType.SOLUTION_TYPE_SEARCH],
                content_config=de.DataStore.ContentConfig.NO_CONTENT))
        op.result(timeout=300)
        print("  created data store:", ds_id)
    except AlreadyExists:
        print("  data store exists:", ds_id)

    doc_client = de.DocumentServiceClient()
    branch = f"{_datastore(project, location, ds_id)}/branches/default_branch"
    documents = [de.Document(id=str(r["id"]),
                             json_data=json.dumps({"title": r.get("title", str(r["id"])),
                                                   "text": r["text"]}))
                 for r in rows]
    imp = doc_client.import_documents(request=de.ImportDocumentsRequest(
        parent=branch,
        inline_source=de.ImportDocumentsRequest.InlineSource(documents=documents),
        reconciliation_mode=de.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL))
    imp.result(timeout=300)
    print(f"  imported {len(documents)} documents")

    eng_client = de.EngineServiceClient()
    try:
        op = eng_client.create_engine(
            parent=parent, engine_id=engine_id,
            engine=de.Engine(
                display_name="Code Assistant Agent Docs Search",
                solution_type=de.SolutionType.SOLUTION_TYPE_SEARCH,
                industry_vertical=de.IndustryVertical.GENERIC,
                data_store_ids=[ds_id],
                search_engine_config=de.Engine.SearchEngineConfig(
                    search_tier=de.SearchTier.SEARCH_TIER_STANDARD)))
        op.result(timeout=300)
        print("  created engine:", engine_id)
    except AlreadyExists:
        print("  engine exists:", engine_id)

    return serving_config(project, location, engine_id)


def _extract_text(doc):
    for attr in ("struct_data", "derived_struct_data"):
        sd = getattr(doc, attr, None)
        if sd:
            d = dict(sd)
            if d.get("text"):
                return str(d["text"])
    if getattr(doc, "json_data", ""):
        try:
            d = json.loads(doc.json_data)
            if d.get("text"):
                return str(d["text"])
        except Exception:
            pass
    return ""


# One client for all searches (a fresh channel per call is slow), and an explicit
# transient-error retry: the GAPIC search method ships with no default retry, which
# made this the only unretried API in the measured hot path.
_search_client = None
_SEARCH_RETRY = core_retry.Retry(predicate=core_retry.if_transient_error,
                                 initial=1.0, maximum=8.0, timeout=60.0)


def search(serving_config_path, query, top_k=5):
    global _search_client
    if _search_client is None:
        _search_client = de.SearchServiceClient()
    resp = _search_client.search(de.SearchRequest(serving_config=serving_config_path,
                                                  query=query, page_size=top_k),
                                 retry=_SEARCH_RETRY, timeout=30.0)
    out = []
    for r in resp:
        t = _extract_text(r.document)
        if t:
            out.append(t)
    return out


def wait_until_searchable(serving_config_path, probe="KeyError", timeout=600, interval=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if search(serving_config_path, probe, top_k=3):
                return True
        except GoogleAPICallError:
            pass
        time.sleep(interval)
    return False


def ensure(project, location, ds_id, engine_id, rows, *, probe="KeyError", timeout=600):
    """Idempotent setup. If the search engine already returns results, reuse it
    (fast path for repeated test runs). Otherwise provision and wait until
    searchable (the Qwiklabs fresh-environment path). Returns the serving config."""
    sc = serving_config(project, location, engine_id)
    try:
        if search(sc, probe, top_k=1):
            print("Agent Search already provisioned; reusing engine", engine_id)
            return sc
    except Exception:
        pass  # engine / data store not present yet -> provision below
    provision(project, location, ds_id, engine_id, rows)
    if not wait_until_searchable(sc, probe, timeout=timeout):
        raise RuntimeError("Agent Search data store not searchable after provisioning")
    return sc


def teardown(project, location, ds_id, engine_id):
    """Delete the engine then the data store (cleanup; the engine references the
    data store, so it must go first)."""
    from google.api_core.exceptions import NotFound
    try:
        de.EngineServiceClient().delete_engine(
            name=f"{_collection(project, location)}/engines/{engine_id}").result(timeout=300)
        print("deleted engine:", engine_id)
    except NotFound:
        print("engine not found:", engine_id)
    try:
        de.DataStoreServiceClient().delete_data_store(
            name=_datastore(project, location, ds_id)).result(timeout=300)
        print("deleted data store:", ds_id)
    except NotFound:
        print("data store not found:", ds_id)


def make_vais_lookup_tool(serving_config_path, ledger, qid_holder, *, agent="DocsAgent", top_k=5):
    """ADK FunctionTool source: query managed Agent Search, bill a vais_query."""
    def lookup(query: str) -> str:
        docs = search(serving_config_path, query, top_k=top_k)
        ledger.add(query_id=qid_holder["v"], agent=agent, call_type="retrieval",
                   model="vais_query", retrieval_queries=1)
        return "\n\n".join(docs) if docs else "No results found."
    lookup.__name__ = "docsagent_lookup"
    lookup.__doc__ = "Search official product docs and git repos for a coding question or error."
    return lookup
