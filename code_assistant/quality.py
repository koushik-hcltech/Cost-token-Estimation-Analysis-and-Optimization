"""Quality scoring via the Gen AI evaluation service.

The judge is an LLM (an autorater). The metric is ours, not a canned template: a
reference-anchored correctness rubric, so the dataset's reference answers participate
in scoring. The notebook's "Scoring quality" section prints CRITERIA and RATING_RUBRIC;
this module is where they are defined, and the service assembles them into the exact
prompt the judge model receives (with {prompt}, {response}, and {reference} filled in
per row).
"""
from vertexai.evaluation import PointwiseMetric, PointwiseMetricPromptTemplate

CRITERIA = {
    "correctness": ("The response's technical claims and recommended fixes agree with the "
                    "reference answer. It contains no claim that contradicts the reference."),
    "completeness": ("The response covers the essential points of the reference answer; for "
                     "multi-part questions, it addresses every part."),
}

RATING_RUBRIC = {
    "5": "Fully correct and complete: agrees with the reference on all key points and covers them all.",
    "4": "Correct with minor omissions: agrees with the reference but misses a small detail.",
    "3": "Partially correct: covers some key points of the reference but misses others.",
    "2": "Mostly incomplete or contains an incorrect claim on a key point.",
    "1": "Wrong: contradicts the reference or fails to answer the question.",
}

CORRECTNESS_METRIC = PointwiseMetric(
    metric="question_answering_correctness",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria=CRITERIA,
        rating_rubric=RATING_RUBRIC,
        input_variables=["prompt", "reference"],
    ),
)


def run_quality_eval(records, project, location):
    "records: [{'query','answer','reference'}] -> mean reference-anchored correctness in [0,1]."
    import math
    import time
    import vertexai
    import pandas as pd
    from vertexai.evaluation import EvalTask

    vertexai.init(project=project, location=location)
    df = pd.DataFrame([{"prompt": r["query"], "response": r["answer"], "reference": r.get("reference", "")}
                       for r in records])
    for attempt in (1, 2):
        try:
            res = EvalTask(dataset=df, metrics=[CORRECTNESS_METRIC]).evaluate()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)   # one retry shields a whole measured run from a transient eval-service error
    summary = getattr(res, "summary_metrics", None) or {}
    mean5 = summary.get("question_answering_correctness/mean")
    # The eval service scores failed judge rows as NaN and averages the rest; if every
    # row failed the mean is NaN (or absent). That must be an error, not a score: the
    # old clamp turned an all-NaN mean into a perfect 1.0.
    if mean5 is None or math.isnan(float(mean5)):
        raise RuntimeError("quality eval returned no scores (all judge rows failed; "
                           "usually a transient service issue). Re-run this cell.")
    return round(max(0.0, min(1.0, (float(mean5) - 1) / 4)), 4)
