#!/usr/bin/env python3
"""
Run an OfficeQA-style OpenAI agent baseline with GPT-5.5 + hosted File Search.

This is the non-oracle, agentic setting: the model receives only the question and
has access to the full Treasury Bulletin corpus through OpenAI's Responses API
`file_search` tool. It does not receive source_docs, source_files, oracle pages,
or the ground-truth answer in the prompt.

The closest public description of Databricks' OpenAI agent baseline says:
  - OpenAI Responses API
  - GPT agent configured with reasoning_effort=high
  - access to File Search and Web Search tools
  - PDFs uploaded to an OpenAI Vector Store
  - a separate parsed-document variant using Databricks-parsed documents

This script supports both:
  - --corpus pdf          raw PDF vector store baseline
  - --corpus transformed parsed-text vector store variant (default here because
                         this repo ships transformed text and it is cheaper to
                         ingest while preserving the same Responses tool path)

Prereqs:
  pip install openai pandas tqdm
  export OPENAI_API_KEY=...

Typical usage:
  # 1) Build or reuse a vector store over the full parsed-text corpus.
  python scripts/run_openai_agentic_file_search.py setup --corpus transformed

  # The setup command prints a vector_store_id. Reuse it for eval:
  python scripts/run_openai_agentic_file_search.py eval --vector-store-id vs_...

  # Raw PDF variant, if treasury_bulletin_pdfs/ contains the full corpus:
  python scripts/run_openai_agentic_file_search.py setup --corpus pdf
  python scripts/run_openai_agentic_file_search.py eval --vector-store-id vs_... --corpus pdf
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reward import extract_final_answer, fuzzy_match_answer  # noqa: E402

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
DEFAULT_IN_RATE = float(os.environ.get("OPENAI_PRICE_INPUT_PER_MTOK", "5.0"))
DEFAULT_OUT_RATE = float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_MTOK", "30.0"))


def init_agents_tracing(args: argparse.Namespace) -> None:
    if not args.enable_openai_tracing:
        return
    try:
        from agents import set_tracing_export_api_key
    except ImportError as e:
        raise SystemExit(
            "OpenAI Agents SDK tracing requested but `agents` is not installed. "
            "Install it with: pip install openai-agents"
        ) from e

    api_key = os.environ.get(args.tracing_api_key_env) or os.environ.get("OPENAI_API_KEY")
    if api_key:
        set_tracing_export_api_key(api_key)


def trace_context(args: argparse.Namespace, row: pd.Series):
    if not args.enable_openai_tracing:
        return nullcontext(None)
    from agents import trace

    trace_id = args.trace_id or f"trace_{uuid.uuid4().hex}"
    metadata = {
        "uid": str(row["uid"]),
        "model": args.model,
        "vector_store_id": args.vector_store_id,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_summary": args.reasoning_summary,
        "max_file_results": args.max_file_results,
        "max_output_tokens": args.max_output_tokens,
        "corpus": args.corpus,
    }
    if args.trace_include_question:
        metadata["question"] = str(row["question"])

    return trace(
        args.trace_workflow_name,
        trace_id=trace_id,
        group_id=args.trace_group_id or str(row["uid"]),
        metadata=metadata,
    )


def custom_trace_span(args: argparse.Namespace, name: str, data: dict[str, Any] | None = None):
    if not args.enable_openai_tracing:
        return nullcontext(None)
    from agents import custom_span

    return custom_span(name, data or {})


def flush_openai_traces(args: argparse.Namespace) -> None:
    if not args.enable_openai_tracing or not args.flush_traces:
        return
    from agents import flush_traces

    flush_traces()


def corpus_dir_for(mode: str) -> Path:
    if mode == "pdf":
        return REPO_ROOT / "treasury_bulletin_pdfs"
    if mode == "transformed":
        return REPO_ROOT / "treasury_bulletins_parsed" / "transformed"
    if mode == "json":
        return REPO_ROOT / "treasury_bulletins_parsed" / "jsons"
    raise ValueError(f"unknown corpus mode: {mode}")


def corpus_glob_for(mode: str) -> str:
    return {"pdf": "*.pdf", "transformed": "*.txt", "json": "*.json"}[mode]


def list_corpus_files(mode: str, corpus_dir: Path, upload_limit: int | None = None) -> list[Path]:
    files = sorted(corpus_dir.glob(corpus_glob_for(mode)))
    if upload_limit:
        files = files[:upload_limit]
    if not files:
        raise FileNotFoundError(
            f"No {corpus_glob_for(mode)} files found in {corpus_dir}. "
            "Extract the corpus first; see README.md."
        )
    return files


def create_vector_store(client: Any, name: str) -> str:
    store = client.vector_stores.create(name=name)
    return store.id


def upload_files(client: Any, vector_store_id: str, files: list[Path], sleep_s: float = 0.0) -> None:
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] uploading {path.name}", flush=True)
        with path.open("rb") as f:
            # OpenAI SDK convenience method: upload, attach to vector store, and poll.
            client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store_id,
                file=f,
            )
        if sleep_s:
            time.sleep(sleep_s)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * DEFAULT_IN_RATE + output_tokens * DEFAULT_OUT_RATE) / 1_000_000.0


def usage_counts(response: Any) -> tuple[int, int, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, None
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = None
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            reasoning_tokens = int(reasoning_tokens)
    return input_tokens, output_tokens, reasoning_tokens


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    return str(value)


def output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                parts.append(str(content_text))
    return "\n".join(parts)


def response_output_items(response: Any) -> list[dict[str, Any]]:
    return [to_jsonable(item) for item in getattr(response, "output", []) or []]


def response_metadata(response: Any) -> dict[str, Any]:
    return {
        "id": getattr(response, "id", None),
        "created_at": getattr(response, "created_at", None),
        "status": getattr(response, "status", None),
        "model": getattr(response, "model", None),
        "incomplete_details": to_jsonable(getattr(response, "incomplete_details", None)),
        "error": to_jsonable(getattr(response, "error", None)),
        "parallel_tool_calls": getattr(response, "parallel_tool_calls", None),
        "tool_choice": to_jsonable(getattr(response, "tool_choice", None)),
        "usage": to_jsonable(getattr(response, "usage", None)),
    }


def reasoning_items(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        summaries = []
        for summary in getattr(item, "summary", []) or []:
            summaries.append(
                {
                    "type": getattr(summary, "type", None),
                    "text": getattr(summary, "text", None),
                }
            )
        items.append(
            {
                "id": getattr(item, "id", None),
                "status": getattr(item, "status", None),
                "summary": summaries,
                "encrypted_content_present": bool(getattr(item, "encrypted_content", None)),
            }
        )
    return items


def content_annotations(response: Any) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                annotations.append(to_jsonable(annotation))
    return annotations


def result_text(result: Any) -> str | None:
    content = getattr(result, "content", None)
    if not content:
        return None
    first = content[0]
    return getattr(first, "text", None)


def file_search_results(item: Any) -> list[Any]:
    # Depending on SDK/API version and include setting, the field may appear as
    # either `results` or `search_results`.
    return list(getattr(item, "results", None) or getattr(item, "search_results", None) or [])


def file_search_items(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "file_search_call":
            continue
        entry: dict[str, Any] = {
            "id": getattr(item, "id", None),
            "status": getattr(item, "status", None),
            "queries": list(getattr(item, "queries", []) or []),
        }
        results = []
        for rank, result in enumerate(file_search_results(item), start=1):
            results.append(
                {
                    "rank": rank,
                    "file_id": getattr(result, "file_id", None),
                    "filename": getattr(result, "filename", None),
                    "score": getattr(result, "score", None),
                    "text": result_text(result),
                }
            )
        if results:
            entry["results"] = results
        items.append(entry)
    return items


def cited_files(response: Any) -> list[dict[str, Any]]:
    seen: set[tuple[str | None, str | None]] = set()
    files: list[dict[str, Any]] = []
    for ann in content_annotations(response):
        if ann.get("type") != "file_citation":
            continue
        key = (ann.get("file_id"), ann.get("filename"))
        if key in seen:
            continue
        seen.add(key)
        files.append(
            {
                "file_id": ann.get("file_id"),
                "filename": ann.get("filename"),
                "citation_index": ann.get("index"),
            }
        )
    return files


def retrieved_files_from_search(response: Any) -> list[dict[str, Any]]:
    seen: set[tuple[str | None, str | None]] = set()
    files: list[dict[str, Any]] = []
    for call in file_search_items(response):
        for result in call.get("results", []):
            key = (result.get("file_id"), result.get("filename"))
            if key in seen:
                continue
            seen.add(key)
            files.append(
                {
                    "file_id": result.get("file_id"),
                    "filename": result.get("filename"),
                    "best_rank": result.get("rank"),
                    "best_score": result.get("score"),
                    "file_search_call_id": call.get("id"),
                }
            )
    return files


def tool_call_items(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if not isinstance(item_type, str) or not item_type.endswith("_call"):
            continue
        calls.append(to_jsonable(item))
    return calls


def summarize_event(event: Any, started_monotonic: float) -> dict[str, Any]:
    event_type = getattr(event, "type", None)
    entry: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "event_type": event_type,
    }

    item = getattr(event, "item", None)
    if item is not None:
        entry["item_id"] = getattr(item, "id", None)
        entry["item_type"] = getattr(item, "type", None)
        entry["item_status"] = getattr(item, "status", None)
        entry["item_name"] = getattr(item, "name", None)
        if getattr(item, "type", None) == "file_search_call":
            entry["queries"] = list(getattr(item, "queries", []) or [])
            result_files = []
            for result in file_search_results(item):
                result_files.append(
                    {
                        "file_id": getattr(result, "file_id", None),
                        "filename": getattr(result, "filename", None),
                        "score": getattr(result, "score", None),
                    }
                )
            if result_files:
                entry["result_files"] = result_files

    output_index = getattr(event, "output_index", None)
    if output_index is not None:
        entry["output_index"] = output_index

    delta = getattr(event, "delta", None)
    if isinstance(delta, str) and delta:
        entry["delta_chars"] = len(delta)

    return entry


def print_event_summary(entry: dict[str, Any]) -> None:
    event_type = entry.get("event_type")
    if event_type in {"response.output_text.delta", "response.reasoning_summary_text.delta"}:
        return

    parts = [
        f"  event +{entry.get('elapsed_seconds', 0):.3f}s",
        str(event_type),
    ]
    if entry.get("item_type"):
        parts.append(f"item={entry['item_type']}")
    if entry.get("item_status"):
        parts.append(f"status={entry['item_status']}")
    if entry.get("item_name"):
        parts.append(f"name={entry['item_name']}")
    if entry.get("queries"):
        parts.append(f"queries={len(entry['queries'])}")
    if entry.get("result_files"):
        filenames = [f.get("filename") for f in entry["result_files"] if f.get("filename")]
        parts.append(f"files={filenames[:5]}")
    print(" ".join(parts), flush=True)


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate limit" in message or "tokens per min" in message


def is_tpm_rate_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return "tokens per min" in message or "tpm" in message


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return float(value)
            except ValueError:
                pass

    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def create_response_with_retries(
    client: Any,
    kwargs: dict[str, Any],
    args: argparse.Namespace,
    event_timeline: list[dict[str, Any]],
    started_monotonic: float,
) -> Any:
    default_wait = args.initial_rate_limit_wait
    for attempt in range(args.max_rate_limit_retries + 1):
        if attempt:
            event_timeline.append(
                {
                    "timestamp_utc": utc_now_iso(),
                    "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                    "event_type": "client.retry.started",
                    "attempt": attempt + 1,
                }
            )

        attempt_data = {
            "attempt": attempt + 1,
            "model": kwargs.get("model"),
            "stream": args.stream,
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        }
        attempt_span = None
        try:
            with custom_trace_span(args, "openai.responses.attempt", attempt_data) as attempt_span:
                if args.stream:
                    with client.responses.stream(**kwargs) as stream:
                        try:
                            for event in stream:
                                event_summary = summarize_event(event, started_monotonic)
                                event_timeline.append(event_summary)
                                if args.print_events:
                                    print_event_summary(event_summary)
                        except Exception as stream_exc:
                            # Streaming can fail mid-flight (including 429 TPM limits). OpenAI does not
                            # support reconnecting to an interrupted SSE stream; the practical recovery
                            # is to start a new Responses request after waiting.
                            stream_error = {
                                "timestamp_utc": utc_now_iso(),
                                "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                                "event_type": "client.stream.error",
                                "attempt": attempt + 1,
                                "error_type": type(stream_exc).__name__,
                                "error": str(stream_exc),
                            }
                            event_timeline.append(stream_error)
                            if attempt_span is not None:
                                attempt_span.set_error(
                                    {
                                        "message": str(stream_exc),
                                        "data": stream_error,
                                    }
                                )
                            raise stream_exc
                        return stream.get_final_response()
                return client.responses.create(**kwargs)
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= args.max_rate_limit_retries:
                event_timeline.append(
                    {
                        "timestamp_utc": utc_now_iso(),
                        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                        "event_type": "client.request.error",
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt_span is not None:
                    attempt_span.set_error(
                        {
                            "message": str(exc),
                            "data": {
                                "attempt": attempt + 1,
                                "error_type": type(exc).__name__,
                            },
                        }
                    )
                raise

            wait_from_api = retry_after_seconds(exc)
            wait_seconds = (wait_from_api if wait_from_api is not None else default_wait) + args.rate_limit_wait_buffer
            max_wait = args.max_rate_limit_wait
            if is_tpm_rate_limit(exc):
                max_wait = max(max_wait, args.max_rate_limit_wait_tpm)
            if is_tpm_rate_limit(exc) and args.tpm_wait_floor > 0:
                wait_seconds = max(wait_seconds, args.tpm_wait_floor)
            wait_seconds = min(wait_seconds, max_wait)
            event_timeline.append(
                {
                    "timestamp_utc": utc_now_iso(),
                    "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                    "event_type": "client.rate_limit.wait",
                    "attempt": attempt + 1,
                    "wait_seconds": round(wait_seconds, 3),
                    "api_retry_after_seconds": wait_from_api,
                    "max_wait_cap_seconds": max_wait,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            with custom_trace_span(
                args,
                "openai.responses.rate_limit_wait",
                {
                    "attempt": attempt + 1,
                    "wait_seconds": round(wait_seconds, 3),
                    "api_retry_after_seconds": wait_from_api,
                    "max_wait_cap_seconds": max_wait,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ):
                pass
            print(
                f"  rate limited on attempt {attempt + 1}; waiting {wait_seconds:.1f}s before retry",
                flush=True,
            )
            time.sleep(wait_seconds)
            default_wait = min(default_wait * 2, max_wait)

    raise RuntimeError("unreachable retry loop exit")


def build_tools(args: argparse.Namespace) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "file_search",
            "vector_store_ids": [args.vector_store_id],
            "max_num_results": args.max_file_results,
        }
    ]
    if not args.no_web_search:
        # Public OfficeQA baseline description gives the GPT agent web search too.
        # The tool type is configurable because OpenAI has renamed preview tools before.
        tools.append({"type": args.web_search_tool})
    return tools


def make_agent_input(question: str) -> list[dict[str, str]]:
    system = (
        "You are an OfficeQA benchmark agent. Answer the user's question using the "
        "Treasury Bulletin corpus available through file_search. Use web search only "
        "for external factual context if needed; do not use it as a substitute for "
        "Treasury Bulletin values. Be careful with fiscal vs calendar years, units, "
        "revisions, table headers, and arithmetic. Return the final answer inside "
        "<FINAL_ANSWER></FINAL_ANSWER> tags."
    )
    user = f"OfficeQA question:\n{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_one_question(client: Any, row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    include = []
    if args.include_search_results:
        # Current Responses docs expose hosted retrieval details through include.
        include.append("output[*].file_search_call.search_results")
    if args.include_encrypted_reasoning:
        # This is encrypted reasoning state, not raw chain-of-thought text.
        include.append("reasoning.encrypted_content")
    reasoning: dict[str, str] = {"effort": args.reasoning_effort}
    if args.reasoning_summary != "none":
        reasoning["summary"] = args.reasoning_summary

    kwargs: dict[str, Any] = {
        "model": args.model,
        "input": make_agent_input(str(row["question"])),
        "tools": build_tools(args),
        "reasoning": reasoning,
        "max_output_tokens": args.max_output_tokens,
    }
    if include:
        kwargs["include"] = include
    if args.tool_choice != "auto":
        kwargs["tool_choice"] = args.tool_choice

    started_monotonic = time.monotonic()
    started_wall = utc_now_iso()
    event_timeline: list[dict[str, Any]] = [
        {
            "timestamp_utc": started_wall,
            "elapsed_seconds": 0.0,
            "event_type": "client.request.started",
        }
    ]

    with custom_trace_span(
        args,
        "officeqa.responses_request",
        {
            "uid": str(row["uid"]),
            "model": args.model,
            "vector_store_id": args.vector_store_id,
            "tools": [tool.get("type") for tool in kwargs["tools"]],
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "reasoning": kwargs["reasoning"],
            "max_output_tokens": kwargs["max_output_tokens"],
            "include": kwargs.get("include", []),
        },
    ):
        response = create_response_with_retries(client, kwargs, args, event_timeline, started_monotonic)

    finished_wall = utc_now_iso()
    elapsed_seconds = time.monotonic() - started_monotonic
    event_timeline.append(
        {
            "timestamp_utc": finished_wall,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "event_type": "client.request.finished",
        }
    )

    raw = output_text(response)
    try:
        pred = extract_final_answer(raw)
    except ValueError:
        pred = raw

    ok, rationale = fuzzy_match_answer(str(row["answer"]), pred, args.tolerance)
    input_tokens, output_tokens, reasoning_tokens = usage_counts(response)
    file_search_calls = file_search_items(response)
    annotations = content_annotations(response)
    retrieved_files = retrieved_files_from_search(response)
    cited = cited_files(response)
    with custom_trace_span(
        args,
        "officeqa.file_search_summary",
        {
            "uid": str(row["uid"]),
            "queries": [
                query
                for call in file_search_calls
                for query in call.get("queries", [])
            ],
            "retrieved_files": retrieved_files,
            "cited_files": cited,
        },
    ):
        pass
    with custom_trace_span(
        args,
        "officeqa.scoring",
        {
            "uid": str(row["uid"]),
            "score": 1.0 if ok else 0.0,
            "match_rationale": rationale,
            "ground_truth": str(row["answer"]) if args.trace_include_sensitive_data else None,
            "prediction": pred if args.trace_include_sensitive_data else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens),
            "elapsed_seconds": elapsed_seconds,
        },
    ):
        pass

    return {
        "uid": row["uid"],
        "question": row["question"],
        "answer": row["answer"],
        "prediction": pred,
        "raw_output": raw,
        "score": 1.0 if ok else 0.0,
        "match_rationale": rationale,
        "model": args.model,
        "vector_store_id": args.vector_store_id,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_summary_setting": args.reasoning_summary,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens),
        "started_at_utc": started_wall,
        "finished_at_utc": finished_wall,
        "elapsed_seconds": elapsed_seconds,
        "request": {
            "input": kwargs["input"],
            "tools": kwargs["tools"],
            "reasoning": kwargs["reasoning"],
            "max_output_tokens": kwargs["max_output_tokens"],
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "include": kwargs.get("include", []),
        },
        "event_timeline": event_timeline,
        "response_metadata": response_metadata(response),
        "response_output_items": response_output_items(response),
        "reasoning_items": reasoning_items(response),
        "content_annotations": annotations,
        "tool_call_items": tool_call_items(response),
        "file_search_calls": file_search_calls,
        "file_search_queries": [
            query
            for call in file_search_calls
            for query in call.get("queries", [])
        ],
        "retrieved_files": retrieved_files,
        "cited_files": cited,
        "response_id": getattr(response, "id", None),
    }


def cmd_setup(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI()
    corpus_dir = args.corpus_dir or corpus_dir_for(args.corpus)
    files = list_corpus_files(args.corpus, corpus_dir, args.upload_limit)
    name = args.name or f"officeqa-{args.corpus}-{int(time.time())}"

    vector_store_id = create_vector_store(client, name)
    print(f"created vector_store_id={vector_store_id}")
    print(f"uploading {len(files)} {args.corpus} files from {corpus_dir}")
    upload_files(client, vector_store_id, files, sleep_s=args.sleep_s)
    print(f"done. Reuse with: --vector-store-id {vector_store_id}")


def cmd_eval(args: argparse.Namespace) -> None:
    from openai import OpenAI

    if not args.vector_store_id:
        raise SystemExit("eval requires --vector-store-id (run setup first)")

    init_agents_tracing(args)
    client = OpenAI()
    df = pd.read_csv(args.csv, dtype=str)
    if args.uid:
        df = df.loc[df["uid"] == args.uid]
        if df.empty:
            raise SystemExit(f"uid not found: {args.uid}")
    if args.limit:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores: list[float] = []
    total_cost = 0.0
    with args.output.open("w", encoding="utf-8") as out:
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"running {row['uid']}...", flush=True)
            try:
                with trace_context(args, row):
                    result = run_one_question(client, row, args)
            finally:
                flush_openai_traces(args)
            scores.append(float(result["score"]))
            total_cost += float(result["estimated_cost_usd"])
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"  score={result['score']} "
                f"tokens={result['input_tokens']}+{result['output_tokens']} "
                f"cost=${result['estimated_cost_usd']:.4f}"
            )
            if args.sleep_between_samples > 0 and i < len(df) - 1:
                time.sleep(args.sleep_between_samples)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nwrote {args.output}")
    print(f"n={len(scores)} mean_score={mean_score:.4f} estimated_cost=${total_cost:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    setup = sub.add_parser("setup", help="create a vector store and upload corpus files")
    setup.add_argument("--corpus", choices=["pdf", "transformed", "json"], default="transformed")
    setup.add_argument("--corpus-dir", type=Path, default=None)
    setup.add_argument("--name", type=str, default=None)
    setup.add_argument("--upload-limit", type=int, default=None, help="testing only; default uploads all files")
    setup.add_argument("--sleep-s", type=float, default=0.0, help="optional delay between uploads")
    setup.set_defaults(func=cmd_setup)

    eval_p = sub.add_parser("eval", help="run OfficeQA questions using an existing vector store")
    eval_p.add_argument("--csv", type=Path, default=REPO_ROOT / "officeqa_pro.csv")
    eval_p.add_argument("--uid", type=str, default=None)
    eval_p.add_argument("--limit", type=int, default=1, help="default runs one sample; use 0 for all")
    eval_p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "openai_agentic_file_search.jsonl")
    eval_p.add_argument("--vector-store-id", type=str, required=True)
    eval_p.add_argument("--corpus", choices=["pdf", "transformed", "json"], default="transformed")
    eval_p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    eval_p.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="high",
    )
    eval_p.add_argument(
        "--reasoning-summary",
        choices=["none", "auto", "concise", "detailed"],
        default="auto",
        help="Requests exposed reasoning summaries; raw chain-of-thought is not exposed by GPT-5.5",
    )
    eval_p.add_argument(
        "--tool-choice",
        choices=["auto", "required"],
        default="auto",
        help="baseline-like default is auto; use required to force at least one tool call",
    )
    eval_p.add_argument("--max-file-results", type=int, default=20)
    eval_p.add_argument("--max-output-tokens", type=int, default=12000)
    eval_p.add_argument("--max-rate-limit-retries", type=int, default=8)
    eval_p.add_argument("--initial-rate-limit-wait", type=float, default=10.0)
    eval_p.add_argument("--max-rate-limit-wait", type=float, default=180.0)
    eval_p.add_argument(
        "--max-rate-limit-wait-tpm",
        type=float,
        default=420.0,
        help="max wait cap for tokens-per-minute (TPM) 429s; can exceed --max-rate-limit-wait",
    )
    eval_p.add_argument(
        "--tpm-wait-floor",
        type=float,
        default=65.0,
        help="minimum wait (seconds) after TPM 429s to reduce immediate re-fail in the same minute window",
    )
    eval_p.add_argument(
        "--rate-limit-wait-buffer",
        type=float,
        default=2.0,
        help="extra seconds added to OpenAI retry-after / 'try again in' waits",
    )
    eval_p.add_argument(
        "--sleep-between-samples",
        type=float,
        default=0.0,
        help="optional pause between CSV rows when running many samples (reduces TPM bursts)",
    )
    eval_p.add_argument("--tolerance", type=float, default=0.01)
    eval_p.add_argument("--no-web-search", action="store_true")
    eval_p.add_argument("--web-search-tool", type=str, default="web_search_preview")
    eval_p.add_argument("--include-search-results", action="store_true")
    eval_p.add_argument(
        "--print-events",
        action="store_true",
        help="print non-delta streaming events live so you can see tool/action progress before final JSONL",
    )
    eval_p.add_argument(
        "--include-encrypted-reasoning",
        action="store_true",
        help="Include encrypted reasoning state for multi-turn/ZDR workflows; this is not readable CoT",
    )
    eval_p.add_argument(
        "--enable-openai-tracing",
        action="store_true",
        help="Upload per-sample custom traces/spans to the OpenAI Traces dashboard via the Agents SDK",
    )
    eval_p.add_argument("--trace-workflow-name", type=str, default="OfficeQA GPT-5.5 agentic eval")
    eval_p.add_argument("--trace-group-id", type=str, default=None)
    eval_p.add_argument(
        "--trace-id",
        type=str,
        default=None,
        help="Optional explicit trace_<32_alphanumeric> id; only useful for a single-row run",
    )
    eval_p.add_argument(
        "--tracing-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Env var used by Agents SDK trace exporter",
    )
    eval_p.add_argument(
        "--trace-include-question",
        action="store_true",
        help="Include question text in trace metadata",
    )
    eval_p.add_argument(
        "--trace-include-sensitive-data",
        action="store_true",
        help="Include ground truth and prediction in scoring span data",
    )
    eval_p.add_argument(
        "--no-flush-traces",
        dest="flush_traces",
        action="store_false",
        help="Do not call agents.flush_traces() after each sample",
    )
    eval_p.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming event capture; tool timings will only have request start/end",
    )
    eval_p.set_defaults(flush_traces=True)
    eval_p.set_defaults(stream=True)
    eval_p.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "limit", None) == 0:
        args.limit = None
    args.func(args)


if __name__ == "__main__":
    main()
