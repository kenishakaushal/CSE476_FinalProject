#  Inference-Time Answering Agent

**Author:** Kenisha Kaushal  
**Course:** CSE 476 – Natural Language Processing

This repository contains a complete implementation of an **inference-time reasoning agent** for generating predictions on the CSE 476 Final Project test dataset. The agent interacts with the class LLM endpoint (`bens_model`) and produces clean, formatted answers for all test questions.

---

##  Agent Architecture

All solving logic is implemented in **`my_agent.py`**, which provides the following components:

### 1. Structured Prompting (Lite Chain-of-Thought)

Each question is processed using a carefully designed system prompt that enforces structured output:

```
You are a careful problem solver. Think briefly step by step,
then give the final answer ONLY as: FINAL ANSWER: <value>
```

This approach minimizes verbose responses while ensuring consistent answer extraction.

### 2. Robust Answer Extraction

The `extract_answer()` function parses model responses by:

- Extracting the substring following `FINAL ANSWER:`
- Trimming whitespace
- Removing trailing punctuation
- Removing leading `+` signs
- Removing internal spaces

This normalization ensures the autograder receives clean, standardized answer strings.

### 3. Retry Logic with Exponential Backoff

To handle transient failures and improve reliability:

- The agent retries failed calls up to **3 times**
- Uses exponential backoff delays (`1s → 2s → 3s`)
- Returns an empty string only after all retries are exhausted

This mechanism significantly improves the completion rate across the entire dataset.

### 4. Concurrent Batch Processing with Autosave

The `generate_all_answers()` function processes questions efficiently:

- **Batch size:** 10 concurrent threads (optimized for server stability)
- **Autosave:** Progress saved after every completed question to `cse_476_final_project_answers_partial.json`

This design prevents data loss during long-running executions and enables real-time progress monitoring.

---

##  How It Works

### Per-Question Processing

For each individual question, the agent:

1. Constructs a structured prompt that enforces `FINAL ANSWER:` format
2. Calls the class LLM endpoint (`bens_model`)
3. Extracts the answer using `extract_answer()`
4. Retries up to 3 times if extraction fails
5. Stores the result in `{"output": <value>}` format
6. Autosaves progress immediately

### Batch Processing

When processing the entire test set:

- Questions are solved **in parallel** (10 concurrent threads)
- A complete partial file is maintained throughout execution
- Progress is logged in real-time with completion percentages
- The final list is returned to `generate_answer_template.py`
- The submission JSON is automatically validated for format compliance

---

##  Setup & Requirements

### Network Access

You must be connected to the **ASU network or VPN** to access the API endpoint:

```
http://10.4.58.53:41701/v1
```

### Dependencies

Install the required Python package:

```bash
pip install requests
```

On macOS, if you encounter permission issues:

```bash
pip install --break-system-packages requests
```

---

##  Usage

### Option 1: Solve a Single Question

```python
from my_agent import run_agent

answer = run_agent("What is 25 * 4?")
print(answer)  # "100"
```

### Option 2: Process the Entire Test Set

```bash
python generate_answer_template.py
```

This will:
- Load all questions from `cse_476_final_project_test_data.json`
- Process them using the agent
- Generate `cse_476_final_project_answers.json` with all predictions
- Validate the output format automatically

---

##  Development & Evaluation

The agent was developed and evaluated through:

- **Development dataset:** `cse476_final_project_dev_data.json`
- Iterative testing on subsamples
- Dry runs of the complete submission pipeline

Key improvements made during development:

- Prompt formatting optimization
- Answer extraction robustness
- Concurrency handling and batch size tuning
- Retry timing and backoff strategies
- Error handling for edge cases

The final implementation reliably generates answers for all questions with high consistency and minimal failures.

---

##  Key Files

| File | Purpose |
|------|---------|
| `my_agent.py` | Core reasoning engine: prompting, extraction, batching, autosave |
| `generate_answer_template.py` | Main entry point; integrates agent with autograder pipeline |
| `cse_476_final_project_answers.json` | Final predictions in submission format |
