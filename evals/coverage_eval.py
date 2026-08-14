"""
Coverage-node offline eval (requirements.md §7.2) — "Push #2".

Runs every example in the LangSmith `coverage-eval` dataset through the REAL
`coverage` node (directly, on hardcoded candidates — no Qdrant search), has an
LLM-judge grade the selected covering set against the ground truth, and records
everything as a LangSmith EXPERIMENT.

    # from the backend/ directory, AFTER uploading the dataset:
    python -m evals.upload_coverage_dataset   # once
    python -m evals.coverage_eval             # each time you want to grade coverage

Only IMPORTS `coverage` from the pipeline (read-only) — it never modifies it.
"""

import asyncio

from dotenv import load_dotenv
from langsmith import aevaluate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.services.matching_graph import coverage

load_dotenv()  # LANGSMITH_* (for the experiment) + ANTHROPIC_API_KEY (for the judge)

DATASET_NAME = "coverage-eval"


# --------------------------------------------------------------------------- #
# 1. Target — run the real coverage node on one case's {requirement, candidates}
# --------------------------------------------------------------------------- #
# We call the node with a hand-built state (NO search node, NO Qdrant). The node
# maps its internal `selected_ids` indices to candidate dicts and returns them as
# `selected_features`, so we read the selection BY NAME (index-agnostic).

async def run_coverage(inputs: dict) -> dict:
    state = {
        "requirement": inputs["requirement"],
        "candidates": inputs["candidates"],
    }
    result = await coverage(state)

    selected = [f.get("name", "") for f in result.get("selected_features", [])]
    return {
        "selected": selected,
        "relevant": len(selected) > 0,        # node returns empty selection when not relevant
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
    }


# --------------------------------------------------------------------------- #
# 2. LLM-judge — grade the selected covering set (fuzzy, by meaning)           #
# --------------------------------------------------------------------------- #

class JudgeVerdict(BaseModel):
    score: float = Field(
        description="Overall coverage quality 0.0 (poor) to 1.0 (perfect): right relevance "
        "decision, every needed feature selected, and no padding."
    )
    missed: list[str] = Field(
        default_factory=list,
        description="Expected features that were NOT selected (needed for coverage but left out).",
    )
    padded: list[str] = Field(
        default_factory=list,
        description="Features that WERE selected but should not have been (distractors / padding).",
    )
    relevant_correct: bool = Field(
        description="True if the relevance decision matched (selected something iff a match was expected).",
    )
    reasoning: str = Field(description="Brief explanation of the score.")


_judge_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=settings.ANTHROPIC_API_KEY,
).with_structured_output(JudgeVerdict)


JUDGE_SYSTEM_PROMPT = """You are grading a "coverage" component. Given a client's requirement and a list of candidate features, it must select the MINIMAL set of candidates that TOGETHER cover the requirement — or select nothing if none are genuinely reusable (judging by underlying logic, not by business domain).

You are given: the requirement, the candidate features, the EXPECTED covering set (ground truth), whether the requirement should have ANY match at all, and the ACTUAL set the component selected.

Judge by MEANING, on three things:
1. Completeness — did the actual set include every feature needed to cover the requirement? A needed feature left out is a MISS.
2. Precision — did the actual set include any candidate that should not be there (a distractor, or padding beyond the minimal covering set)? Each such extra is PADDING.
3. Relevance decision — if no candidate should match, the actual set must be EMPTY; if a match was expected, it must be NON-EMPTY.

Give an overall score from 0.0 to 1.0: 1.0 means the actual set matches the expected covering set with the correct relevance decision, no misses, and no padding. Deduct for each miss and each padded feature, and deduct heavily if the relevance decision itself is wrong. Be fair but strict."""


def _candidates_text(candidates: list) -> str:
    return "\n".join(f"- {c.get('name', '')}: {c.get('description', '')}" for c in candidates)


async def coverage_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    requirement = inputs["requirement"]
    candidates = inputs["candidates"]
    actual = outputs.get("selected", [])
    expected = reference_outputs.get("expected_selected", [])
    expected_relevant = reference_outputs.get("expected_relevant", bool(expected))

    actual_text = "\n".join(f"- {name}" for name in actual) or "(none selected)"
    expected_text = "\n".join(f"- {name}" for name in expected) or "(none — no match expected)"

    user = (
        f"REQUIREMENT:\n"
        f"name: {requirement.get('name', '')}\n"
        f"domain: {requirement.get('domain', '')}\n"
        f"description: {requirement.get('description', '')}\n\n"
        f"CANDIDATE FEATURES:\n{_candidates_text(candidates)}\n\n"
        f"SHOULD ANY CANDIDATE MATCH? {'YES' if expected_relevant else 'NO'}\n\n"
        f"EXPECTED COVERING SET (ground truth):\n{expected_text}\n\n"
        f"ACTUAL SELECTED SET:\n{actual_text}"
    )

    verdict: JudgeVerdict = await _judge_llm.ainvoke(
        [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user)]
    )

    comment = (
        f"selected={actual} | expected={expected} | "
        f"missed={verdict.missed} padded={verdict.padded} "
        f"rel_ok={verdict.relevant_correct} :: {verdict.reasoning}"
    )
    return {"key": "coverage_quality", "score": verdict.score, "comment": comment}


# --------------------------------------------------------------------------- #
# 3. Tie it together — aevaluate creates the Experiment                        #
# --------------------------------------------------------------------------- #

async def main() -> None:
    await aevaluate(
        run_coverage,
        data=DATASET_NAME,
        evaluators=[coverage_quality],
        experiment_prefix="coverage",
        max_concurrency=2,
    )
    print(
        "Eval complete. Open LangSmith -> Datasets & Experiments -> "
        f"'{DATASET_NAME}' -> the newest 'coverage-...' experiment to see the scores."
    )


if __name__ == "__main__":
    asyncio.run(main())
