from quiz_ingest.llm.vllm_client import _parse_json_array_batch


def test_parses_clean_json_array():
    raw = '[{"index": 0, "question": "Q1"}, {"index": 1, "question": "Q2"}]'
    items, failures = _parse_json_array_batch(raw, expected_len=2)
    assert failures == []
    assert items[0]["question"] == "Q1"
    assert items[1]["question"] == "Q2"


def test_tolerates_markdown_fence_and_prose():
    raw = 'Sure, here is the JSON:\n```json\n[{"index": 0, "question": "Q1"}]\n```\nDone.'
    items, failures = _parse_json_array_batch(raw, expected_len=1)
    assert failures == []
    assert items[0]["question"] == "Q1"


def test_partial_parse_failure_marks_missing_indices():
    # Model only returned 1 of 3 expected items -- the other 2 must be
    # reported as failures, NOT silently defaulted to a real-looking dict.
    raw = '[{"index": 0, "question": "Q1"}]'
    items, failures = _parse_json_array_batch(raw, expected_len=3)
    assert failures == [1, 2]
    assert items[1] == {}
    assert items[2] == {}


def test_total_parse_failure_marks_all_indices():
    raw = "I cannot help with that."
    items, failures = _parse_json_array_batch(raw, expected_len=2)
    assert failures == [0, 1]
    assert all(item == {} for item in items)
