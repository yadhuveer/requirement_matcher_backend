"""
Classify-node offline eval (requirements.md §7.2) — "Push #2".

Runs every example in the LangSmith `classify-eval` dataset through the REAL
`classify` node (directly, on a hardcoded selected feature set — no search, no
coverage) and scores by EXACT LABEL MATCH: predicted status == expected status.
No LLM-judge — the output is a binary label, so we just compare.

    # from the backend/ directory, AFTER uploading the dataset:
    python -m evals.upload_classify_dataset   # once
    python -m evals.classify_eval             # each time you want to grade classify

Only IMPORTS `classify` from the pipeline (read-only) — it never modifies it.
"""

import asyncio

from dotenv import load_dotenv
from langsmith import aevaluate

from app.services.matching_graph import classify

load_dotenv()  # LANGSMITH_* for the experiment

DATASET_NAME = "classify-eval"


# --------------------------------------------------------------------------- #
# 1. Target — run the real classify node on one case's {requirement, features} #
# --------------------------------------------------------------------------- #
# We hand classify the ALREADY-selected covering set directly (no search, no
# coverage). It returns {"status": "exact_match" | "needs_modification"}.

async def run_classify(inputs: dict) -> dict:
    state = {
        "requirement": inputs["requirement"],
        "selected_features": inputs["selected_features"],
    }
    result = await classify(state)
    return {"status": result.get("status")}


# --------------------------------------------------------------------------- #
# 2. Evaluator — exact label match (no LLM)                                    #
# --------------------------------------------------------------------------- #

def classify_correct(outputs: dict, reference_outputs: dict) -> dict:
    predicted = outputs.get("status")
    expected = reference_outputs.get("expected_status")
    correct = predicted == expected
    return {
        "key": "classify_correct",
        "score": 1.0 if correct else 0.0,
        "comment": f"predicted={predicted} | expected={expected}",
    }


# --------------------------------------------------------------------------- #
# 3. Tie it together — aevaluate creates the Experiment                        #
# --------------------------------------------------------------------------- #

async def main() -> None:
    await aevaluate(
        run_classify,
        data=DATASET_NAME,
        evaluators=[classify_correct],
        experiment_prefix="classify",
        max_concurrency=2,
    )
    print(
        "Eval complete. Open LangSmith -> Datasets & Experiments -> "
        f"'{DATASET_NAME}' -> the newest 'classify-...' experiment to see the scores."
    )


if __name__ == "__main__":
    asyncio.run(main())
