"""Tiny data helpers (plumbing)."""
import json


def load_jsonl(path):
    "Read a JSON-lines file into a list of dicts."
    return [json.loads(line) for line in open(path) if line.strip()]
