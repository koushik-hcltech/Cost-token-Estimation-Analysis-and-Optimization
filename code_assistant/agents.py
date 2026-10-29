"""Code Assistant Agent multi-agent (ADK): Coordinator consults DocsAgent + CommunityAgent
as tools, then synthesizes a cited answer.

The coordinator uses AgentTool(sub_agent) so control returns to it for final
synthesis (unlike sub_agents transfer, which hands control away). Each agent's
after_model_callback records its own token usage into the shared UsageLedger,
attributed by agent name; retrieval/embedding costs are recorded inside the
tool functions (see retrieval.make_lookup_tool).
"""
import inspect
import asyncio
import threading
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from .retrieval import make_lookup_tool

# Transient API errors (408/429/5xx) retry with the SDK's exponential-backoff policy.
# The google-genai client defaults to a single attempt unless retry options are set,
# so every agent model call opts in here.
_RETRY = types.HttpRetryOptions()


def _model(name):
    "Wrap a model name so its calls carry the retry policy."
    return Gemini(model=name, retry_options=_RETRY)

# Every agent runs under a generation ceiling (max_output_tokens). The cap counts
# thoughts and answer together; typical calls for this agent setup and sample dataset
# sit far below it: sub-agent calls total 150 to 800 tokens (they bill no thought
# tokens here), coordinator calls 850 to 1,500 including thoughts. The ceilings bound
# an occasional unbounded response that generates until it reaches the output limit
# (observed: single calls reaching ~65k tokens, ~$0.59 and ~5 minutes each at Flash
# rates). A deliberate learner-set max_output_tokens replaces the coordinator's ceiling.
SUBAGENT_MAX_OUTPUT = 2048
COORDINATOR_MAX_OUTPUT = 3072

# Gemini 3 family end to end, served from the global endpoint.
DEFAULT_MODELS = {
    "coordinator": "gemini-3.1-pro-preview",
    "docs": "gemini-3.5-flash",
    "community": "gemini-3.5-flash",
    "embedding": "gemini-embedding-001",
}


def _model_cb(ledger, agent_name, model, qid_holder):
    """ADK after_model_callback that records one model billing event."""
    def cb(callback_context, llm_response):
        um = getattr(llm_response, "usage_metadata", None)
        if um is not None:
            # prompt_token_count INCLUDES the cached portion; record the two buckets
            # disjointly so the cost model prices each token exactly once.
            prompt = getattr(um, "prompt_token_count", 0) or 0
            cached = getattr(um, "cached_content_token_count", 0) or 0
            ledger.add(query_id=qid_holder["v"], agent=agent_name, call_type="model", model=model,
                       input_tokens=max(0, prompt - cached),
                       cached_tokens=cached,
                       output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                       thought_tokens=getattr(um, "thoughts_token_count", 0) or 0)
        return None  # do not modify the response
    return cb


def build_code_assistant(*, models, ledger, qid_holder, project, location,
                    collections=None, docs_serving_config=None,
                    top_k=5, max_output_tokens=None, thinking_level=None):
    """Assemble the Code Assistant Agent coordinator.

    DocsAgent backend: managed **Agent Search** when ``docs_serving_config`` is
    given, otherwise the local Chroma stand-in ``collections['docs']``.
    CommunityAgent always uses the local Chroma vector store ``collections['community']``.
    ``thinking_level`` (e.g. "MINIMAL"/"LOW"/"MEDIUM"/"HIGH") pins the coordinator's
    reasoning effort; None keeps the model's default (and the pre-V2 behavior).
    ``max_output_tokens`` caps the coordinator's final answer; None applies the
    COORDINATOR_MAX_OUTPUT safety ceiling rather than leaving generation unbounded.
    """
    if docs_serving_config:
        from .vais import make_vais_lookup_tool
        docs_tool = make_vais_lookup_tool(docs_serving_config, ledger, qid_holder,
                                          agent="DocsAgent", top_k=top_k)
    else:
        docs_tool = make_lookup_tool(collections["docs"], models["embedding"], project, location,
                                     ledger, qid_holder, agent="DocsAgent", top_k=top_k, record_as="vais")
    comm_tool = make_lookup_tool(collections["community"], models["embedding"], project, location,
                                 ledger, qid_holder, agent="CommunityAgent", top_k=top_k, record_as="embedding")

    sub_cfg = types.GenerateContentConfig(max_output_tokens=SUBAGENT_MAX_OUTPUT)
    docs_agent = Agent(
        name="DocsAgent", model=_model(models["docs"]),
        description="Finds answers in official product docs and git repositories.",
        instruction="Call docsagent_lookup with the user's problem, then answer strictly from the results. Be concise.",
        tools=[FunctionTool(docs_tool)],
        generate_content_config=sub_cfg,
        after_model_callback=_model_cb(ledger, "DocsAgent", models["docs"], qid_holder))

    community_agent = Agent(
        name="CommunityAgent", model=_model(models["community"]),
        description="Finds similar prior Stack Overflow discussions.",
        instruction="Call communityagent_lookup with the user's problem, then summarize the most relevant prior discussion. Be concise.",
        tools=[FunctionTool(comm_tool)],
        generate_content_config=sub_cfg,
        after_model_callback=_model_cb(ledger, "CommunityAgent", models["community"], qid_holder))

    cfg_kwargs = {"max_output_tokens": max_output_tokens or COORDINATOR_MAX_OUTPUT}
    if thinking_level:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    coordinator = Agent(
        name="Coordinator", model=_model(models["coordinator"]),
        description="Answers developer coding questions with citations.",
        instruction=(
            "You are Code Assistant Agent, a coding assistant. For the user's coding question or error message, "
            "use the DocsAgent and CommunityAgent tools to gather grounding, then synthesize ONE concise "
            "answer that cites whether it came from official docs or community discussion. "
            "Call a tool only when it helps."),
        tools=[AgentTool(agent=docs_agent), AgentTool(agent=community_agent)],
        after_model_callback=_model_cb(ledger, "Coordinator", models["coordinator"], qid_holder),
        generate_content_config=cfg)
    return coordinator


def _resolve(maybe_coro):
    """Run a coroutine to completion whether or not an event loop is already
    running (notebooks run their own loop, so plain asyncio.run() would fail)."""
    if not inspect.iscoroutine(maybe_coro):
        return maybe_coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(maybe_coro)            # no running loop (scripts)
    import concurrent.futures                      # running loop (notebooks)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, maybe_coro).result()


# ADK drives the agent on a helper thread, and its sync runner swallows exceptions there:
# the run's event generator just ends (runners.py puts a None sentinel in a finally), so a
# failed run would silently look like an empty answer. Keep the one-line warning for runs
# that die AFTER yielding an answer (e.g. a small model hallucinating a tool name), but
# stash the exception so answer_query can raise when the run produced no answer at all.
_default_thread_hook = threading.excepthook
# Written by the excepthook (on ADK's worker thread), consumed by answer_query (on the
# caller's thread) strictly AFTER Runner.run() has joined that worker, so the pair is
# sequenced, not concurrent. Queries in this lab run one at a time by design; a plain
# dict is correct here, and thread-local storage would hide the error from the reader.
_last_run_error = {}


def _quiet_adk_thread_errors(args):
    if "_asyncio_thread_main" in (getattr(args.thread, "name", "") or ""):
        _last_run_error["exc"] = args.exc_value or args.exc_type()
        first = str(args.exc_value).splitlines()[0] if args.exc_value else args.exc_type.__name__
        print(f"[run warning] agent run ended early: {first}")
        return
    _default_thread_hook(args)


threading.excepthook = _quiet_adk_thread_errors


def answer_query(coordinator, query, query_id, qid_holder, *, app_name="code_assistant"):
    """Run one query end-to-end; return the final answer text. Sets qid_holder so
    callbacks attribute usage to this query_id. Raises if the run died without
    producing an answer (the sync runner never raises on its own, see above)."""
    from google.adk.runners import InMemoryRunner
    qid_holder["v"] = query_id
    _last_run_error.pop("exc", None)
    runner = InMemoryRunner(agent=coordinator, app_name=app_name)
    _resolve(runner.session_service.create_session(app_name=app_name, user_id="lab", session_id=query_id))
    msg = types.Content(role="user", parts=[types.Part(text=query)])
    final = ""
    for ev in runner.run(user_id="lab", session_id=query_id, new_message=msg):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "text", None):
                    final = p.text
    if not final.strip():
        exc = _last_run_error.pop("exc", None)
        if exc is not None:
            raise RuntimeError(f"agent run failed before producing an answer: {exc}") from exc
    return final
