"""In-memory usage ledger: one record per billable event.

Plumbing on purpose. The *pricing* stays in the notebook (it is the lab's cost
model); the ledger receives that cost function at construction and never
hardcodes a rate. Agent callbacks and retrieval tools call ``add(...)``; the
notebook reads ``cost_per_query()`` and ``cost_breakdown()``.
"""
from collections import defaultdict


class UsageLedger:
    def __init__(self, cost_fn):
        self._cost_fn = cost_fn        # notebook-defined cost_of(record) -> USD
        self.records = []

    def add(self, **rec):
        for k in ("input_tokens", "cached_tokens", "output_tokens", "thought_tokens",
                  "retrieval_queries", "embed_tokens"):
            rec.setdefault(k, 0)
        rec["cost_usd"] = self._cost_fn(rec)
        self.records.append(rec)
        return rec

    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def total_cost(self):
        return round(sum(r["cost_usd"] for r in self.records), 12)

    def cost_per_query(self):
        totals = defaultdict(float)
        for r in self.records:
            qid = r.get("query_id", "_")
            if qid == "__index__":
                continue   # one-time index build, not a query
            totals[qid] += r["cost_usd"]
        return round(sum(totals.values()) / len(totals), 12) if totals else 0.0

    def cost_breakdown(self):
        "Total USD per agent/component."
        out = {}
        for r in self.records:
            a = r.get("agent", "unknown")
            out[a] = round(out.get(a, 0.0) + r["cost_usd"], 12)
        return out

    def reset(self):
        self.records.clear()
