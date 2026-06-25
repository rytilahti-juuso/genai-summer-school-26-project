from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

LABEL_TAG = re.compile(r"<LABEL>\s*([^<]+?)\s*</LABEL>", re.IGNORECASE)
REASONING_TAG = re.compile(
    r"<REASONING>\s*([^<]+?)\s*</REASONING>",
    re.IGNORECASE,
)
FALLBACK_LABEL = "Not generated"


def create_default_llm():
    """Create the project's configured LLM connection only when requested."""
    from llm_api_connection import LLM_connection
    from secret_key import local_key2

    return LLM_connection("llama4:latest", local_key2)


def get_tag(output: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(output)
    return match.group(1).strip() if match else None


def _save_checkpoint(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def label_clusters(
    llm: Any,
    clusters: Sequence[dict[str, Any]],
    checkpoint_path: str | Path | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate a title and justification while preserving input cluster IDs."""
    results: list[dict[str, Any]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        prompt_text = {
            "task": "Analyze the cluster and create a category label.",
            "description": (
                "Explain the shared topic briefly, then provide one short, concise "
                "category title. Put the title inside <LABEL> tags and the "
                "justification inside <REASONING> tags. Output nothing else."
            ),
            "input": cluster,
        }
        generated_label = FALLBACK_LABEL
        justification = "LLM labeling failed."

        try:
            if llm is None:
                raise RuntimeError("LLM connection is unavailable.")
            output = llm.invoke(json.dumps(prompt_text, ensure_ascii=False))
            parsed_label = get_tag(output, LABEL_TAG)
            parsed_reasoning = get_tag(output, REASONING_TAG)
            if not parsed_label or not parsed_reasoning:
                missing = []
                if not parsed_label:
                    missing.append("LABEL")
                if not parsed_reasoning:
                    missing.append("REASONING")
                justification = (
                    "Malformed LLM response: missing "
                    + " and ".join(missing)
                    + " tag."
                )
            else:
                generated_label = parsed_label
                justification = parsed_reasoning
        except Exception as error:
            justification = f"LLM labeling failed: {type(error).__name__}: {error}"

        result = {
            **cluster,
            "cluster_id": cluster_id,
            "generated_label": generated_label,
            "justification": justification,
        }
        results.append(result)
        if checkpoint is not None:
            _save_checkpoint(results, checkpoint)
        if on_result is not None:
            on_result(result)

    return results

