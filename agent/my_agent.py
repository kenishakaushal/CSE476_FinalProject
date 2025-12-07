#!/usr/bin/env python3
"""
`generate_answer_template.py` will import and call
`generate_all_answers(questions)` from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests  # for API calls

#Config

API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"

PARTIAL_PATH = Path("cse_476_final_project_answers_partial.json")

SYSTEM_PROMPT = (
    "You are a careful problem solver. Think briefly step by step, "
    "then give the final answer ONLY as: FINAL ANSWER: <value>"
)

#Model call helpers


def call_model_chat_completions(
    messages: List[Dict[str, str]],
    model: str = MODEL,
    temperature: float = 0.3,
    timeout: int = 60,
) -> Dict[str, Optional[str]]:
    """Simple wrapper for the course chat endpoint."""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {"ok": True, "text": text}
        return {"ok": False, "text": None}
    except Exception:
        return {"ok": False, "text": None}


def ask_solver(question_text: str) -> str:
    """Ask the model once and return its raw output text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Solve the question below.\n\n"
                f"{question_text}\n\n"
                "End with 'FINAL ANSWER: <value>'."
            ),
        },
    ]
    resp = call_model_chat_completions(messages)
    return resp.get("text") or ""


def extract_answer(raw_text: str) -> Optional[str]:
    """Extract whatever follows 'FINAL ANSWER:' and normalize a bit."""
    if not raw_text or "FINAL ANSWER:" not in raw_text:
        return None
    ans = raw_text.split("FINAL ANSWER:")[-1].strip()
    # Small cleanup: remove trailing dot, leading '+', and spaces
    ans = ans.rstrip(".").lstrip("+").replace(" ", "")
    return ans or None


def run_agent(question_text: str, max_retries: int = 3) -> str:
    """One full agent run for a single question with retry logic."""
    for attempt in range(max_retries):
        raw = ask_solver(question_text)
        answer = extract_answer(raw)
        if answer:  # If we got a valid answer, return it
            return answer.strip()
        # If no answer and not last attempt, wait a bit and retry
        if attempt < max_retries - 1:
            import time
            time.sleep(1 * (attempt + 1))  # Exponential backoff: 1s, 2s, 3s
    return ""  # Return empty if all retries failed


#Batch orchestration


def generate_all_answers(
    questions: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Main entry point used by generate_answer_template.build_answers().

    Takes the list of question objects and returns a list of
    { "output": <answer_string> } dicts of the same length.
    """
    total = len(questions)
    answers: List[Optional[Dict[str, str]]] = [None] * total  # preserve order

    BATCH_SIZE = 10  # how many questions to process concurrently 
    save_lock = Lock()  # for thread-safe partial saves

    def save_answers() -> None:
        """Thread-safe autosave function to PARTIAL_PATH."""
        with save_lock:
            completed_answers = [
                a if a is not None else {"output": ""} for a in answers
            ]
            with PARTIAL_PATH.open("w") as fp:
                json.dump(completed_answers, fp, ensure_ascii=False, indent=2)

    def get_progress_info():
        completed = sum(1 for a in answers if a is not None)
        remaining = total - completed
        percentage = (completed / total * 100) if total > 0 else 0.0
        return completed, remaining, percentage

    print("\n" + "=" * 60, flush=True)
    print(f"Starting to process {total} questions", flush=True)
    print(
        f"Processing in batches of {BATCH_SIZE} questions concurrently",
        flush=True,
    )
    print("=" * 60 + "\n", flush=True)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_questions = questions[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        print(
            f"\n[Batch {batch_start // BATCH_SIZE + 1}] "
            f"Processing questions {batch_start + 1}-{batch_end}/{total}",
            flush=True,
        )
        print(
            f"  Submitting {len(batch_questions)} questions for concurrent processing...\n",
            flush=True,
        )

        # Thread pool for this batch
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_idx = {
                executor.submit(
                    run_agent,
                    q.get("input", ""),
                ): idx
                for idx, q in zip(batch_indices, batch_questions)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    final = future.result().strip()
                    answers[idx] = {"output": final}

                    # Autosave
                    save_answers()

                    completed, remaining, percentage = get_progress_info()
                    print(
                        f"  ✓ Solved question {idx + 1}/{total} | "
                        f"Progress: {completed}/{total} ({percentage:.1f}%) | "
                        f"Remaining: {remaining}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"  ✗ Question {idx + 1} generated an exception: {exc}",
                        flush=True,
                    )
                    answers[idx] = {"output": ""}
                    save_answers()
                    completed, remaining, percentage = get_progress_info()
                    print(
                        f"  ⚠ Question {idx + 1} saved with empty answer | "
                        f"Progress: {completed}/{total} ({percentage:.1f}%) | "
                        f"Remaining: {remaining}",
                        flush=True,
                    )

    final_answers: List[Dict[str, str]] = [
        a if a is not None else {"output": ""} for a in answers
    ]

    print("\n" + "=" * 60, flush=True)
    print(f"Completed processing all {total} questions!", flush=True)
    print(
        f"Final progress: {len(final_answers)}/{total} (100.0%)",
        flush=True,
    )
    print("=" * 60 + "\n", flush=True)

    return final_answers
