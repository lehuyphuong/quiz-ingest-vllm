"""
generation/schemas.py

Schema strings injected literally into prompts (this repo relies on
prompt-described JSON + tolerant parsing in llm/vllm_client.py, not
vLLM's guided_json/grammar-constrained decoding -- see README "Structured
output" section for the trade-off: guided decoding forces every TOKEN to
stay syntactically valid but cannot force a model to actually finish an
open string field correctly, which is exactly the failure mode the
index-based schema below avoids at the source instead).

Every schema here uses an INDEX (0-based integer) to refer back to the
item it's about, never an open-ended string ID -- a small/weak model has a
much easier time closing a short integer field correctly than an open
string it doesn't understand the purpose of. Keep this convention for any
new schema added to this file.
"""

QUESTION_SCHEMA_HINT = (
    '{"index": <0-based int, matching the item number>, '
    '"question": <string>, '
    '"correct_answer": <string, 1-2 sentences>, '
    '"supporting_fact": <string, the exact fact from the context this is grounded in>}'
)

DISTRACTOR_SCHEMA_HINT = (
    '{"index": <0-based int, matching the item number>, '
    '"distractors": [\n'
    '  {"text": <string>, "type": "near_miss"},\n'
    '  {"text": <string>, "type": "misconception"},\n'
    '  {"text": <string>, "type": "plausible_unrelated"}\n'
    "]}"
)

RERANK_SCHEMA_HINT = '{"index": <0-based int>, "score": <float 0-1>}'

NON_CONVERSATIONAL_SYSTEM_INSTRUCTION = (
    "You are a batch JSON-generation function, not a conversational assistant. "
    "Never ask clarifying questions, never apologize, never add commentary "
    "outside the requested JSON. If information is missing, make the most "
    "reasonable choice and proceed."
)
