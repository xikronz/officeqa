#!/usr/bin/env python3
"""
Single-sample OfficeQA eval in the *oracle-page LLM* setting (no tools / no agent).

Loads only the JSON page(s) referenced by `?page=` in `source_docs`, renders them
like the page-level transform (Markdown tables + text), sends one chat completion
to OpenAI, then prints token usage and an approximate USD cost.

Prerequisites
-------------
- ``pip install openai pandas``
- ``export OPENAI_API_KEY=...``
- Oracle page text from either:
  - per-page files ``treasury_bulletins_parsed/transformed_page_level/{stem}_{page}.txt``
    (from ``transform_files_page_level.py --split-files``), or
  - parsed JSON under ``treasury_bulletins_parsed/jsons/`` (run
    ``treasury_bulletins_parsed/unzip.py`` if you have not extracted the corpus).

Usage
-----
  cd /path/to/officeqa
  python scripts/measure_llm_oracle_cost.py
  python scripts/measure_llm_oracle_cost.py --uid UID0001
  OPENAI_MODEL=gpt-5.5-2026-04-23 python scripts/measure_llm_oracle_cost.py

Default cost rates (override with env vars) are for **gpt-5.5** as of the model
card; update ``OPENAI_PRICE_INPUT_PER_MTOK`` / ``OPENAI_PRICE_OUTPUT_PER_MTOK``
if pricing changes — the API returns token counts, not dollars.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TRANSFORM_SCRIPTS = REPO_ROOT / "treasury_bulletins_parsed" / "transform_scripts"
if str(_TRANSFORM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRANSFORM_SCRIPTS))

import transform_files_page_level as _page  # noqa: E402

from reward import extract_final_answer, fuzzy_match_answer  # noqa: E402


def _parse_oracle_pairs(source_docs: str, source_files: str) -> list[tuple[str, int]]:
    """Return (bulletin_stem, page_id) aligned with each oracle URL."""
    urls = [u.strip() for u in str(source_docs).splitlines() if u.strip()]
    files = [f.strip() for f in str(source_files).splitlines() if f.strip()]
    if not urls:
        raise ValueError("source_docs is empty")
    if not files:
        raise ValueError("source_files is empty")

    pairs: list[tuple[str, int]] = []
    for i, url in enumerate(urls):
        m = re.search(r"[?&]page=(\d+)", url)
        if not m:
            raise ValueError(f"No page= query in source_docs URL: {url}")
        page_id = int(m.group(1))
        fname = files[i] if i < len(files) else files[-1]
        stem = Path(fname).stem
        pairs.append((stem, page_id))
    return pairs


def _render_oracle_context(
    json_dir: Path,
    page_level_dir: Path,
    pairs: Iterable[tuple[str, int]],
) -> str:
    """Concatenate markdown for each (bulletin, page) oracle slice."""
    chunks: list[str] = []
    cache: dict[str, dict[int, list[str]]] = {};

    for stem, page_id in pairs:
        split_txt = page_level_dir / f"{stem}_{page_id}.txt"
        if split_txt.is_file():
            body = split_txt.read_text(encoding="utf-8").strip()
            chunks.append(f"### Document `{stem}` — oracle page {page_id}\n\n{body}")
            continue

        if stem not in cache:
            json_path = json_dir / f"{stem}.json"
            if not json_path.is_file():
                raise FileNotFoundError(
                    f"Missing oracle text for {stem} page {page_id}.\n"
                    f"Tried: {split_txt}\n"
                    f"And:   {json_path}\n"
                    "Extract jsons (treasury_bulletins_parsed/unzip.py) or generate "
                    "transformed_page_level splits (transform_files_page_level.py --split-files)."
                )
            cache[stem] = _page.process_file_by_pages(str(json_path))

        page_lines = cache[stem].get(page_id)
        if not page_lines:
            avail = sorted(cache[stem].keys())
            preview = avail if len(avail) <= 30 else avail[:20] + ["…"]
            raise KeyError(
                f"No elements on page_id={page_id} in {stem}.json "
                f"(available page_ids: {preview})"
            )

        body = "\n".join(page_lines).strip()
        chunks.append(f"### Document `{stem}` — oracle page {page_id}\n\n{body}")

    return "\n\n---\n\n".join(chunks)


def _estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    in_per_mtok: float,
    out_per_mtok: float,
) -> float:
    return (prompt_tokens * in_per_mtok + completion_tokens * out_per_mtok) / 1_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "officeqa_pro.csv",
        help="OfficeQA CSV (default: officeqa_pro.csv)",
    )
    parser.add_argument("--uid", type=str, default=None, help="Question uid (default: first row)")
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=REPO_ROOT / "treasury_bulletins_parsed" / "jsons",
        help="Directory with treasury_bulletin_*.json",
    )
    parser.add_argument(
        "--page-level-dir",
        type=Path,
        default=REPO_ROOT / "treasury_bulletins_parsed" / "transformed_page_level",
        help="Optional per-page .txt files: {stem}_{page}.txt (checked before JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-5.5"),
        help="Chat model id (default: gpt-5.5 or OPENAI_MODEL)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for reward.fuzzy_match_answer (default: 0.01 = 1%%)",
    )
    args = parser.parse_args()

    in_rate = float(os.environ.get("OPENAI_PRICE_INPUT_PER_MTOK", "5.0"))
    out_rate = float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_MTOK", "30.0"))

    df = pd.read_csv(args.csv, dtype=str)
    if args.uid is not None:
        row = df.loc[df["uid"] == args.uid]
        if row.empty:
            raise SystemExit(f"uid not found: {args.uid}")
        row = row.iloc[0]
    else:
        row = df.iloc[0]

    pairs = _parse_oracle_pairs(row["source_docs"], row["source_files"])
    context = _render_oracle_context(args.json_dir, args.page_level_dir, pairs)

    system = (
        "You answer questions using ONLY the provided Treasury Bulletin excerpts "
        "(oracle pages). Do not use web search or outside knowledge beyond plain reasoning, make sure to think step by step and show your work. "
        "Put only the final answer inside the tags "
        "<FINAL_ANSWER></FINAL_ANSWER> so it can be scored automatically."
    )
    user = f"Question:\n{row['question']}\n\nRelevant bulletin excerpts:\n\n{context}"

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Install the OpenAI SDK: pip install openai") from e

    client = OpenAI()
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    usage = resp.usage
    if usage is None:
        raise SystemExit("API response missing usage; cannot estimate cost.")

    pt = usage.prompt_tokens
    ct = usage.completion_tokens
    cost = _estimate_cost_usd(pt, ct, in_rate, out_rate)

    raw = resp.choices[0].message.content or ""
    try:
        pred = extract_final_answer(raw)
    except ValueError:
        pred = raw

    ok, rationale = fuzzy_match_answer(row["answer"], pred, args.tolerance)
    score = 1.0 if ok else 0.0

    print("=== OfficeQA oracle-page LLM sample ===")
    print(f"uid:        {row['uid']}")
    print(f"model:      {args.model}")
    print(f"oracle:     {pairs}")
    print(f"tokens:     prompt={pt}  completion={ct}  total={pt + ct}")
    print(
        f"cost (est): ${cost:.6f}  "
        f"(using ${in_rate}/M in, ${out_rate}/M out — set OPENAI_PRICE_*_PER_MTOK to update)"
    )
    print(f"score @ tol={args.tolerance}: {score}")
    print(f"match note: {rationale}")
    print(f"ground_truth: {row['answer']}")
    print(f"prediction:   {pred[:500]}{'…' if len(pred) > 500 else ''}")


if __name__ == "__main__":
    main()
