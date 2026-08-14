"""
Extraction-agent offline eval runner (requirements.md §7.2) — "Push #2".

Runs every example in the LangSmith `extraction-eval` dataset through the real
extraction agent, has an LLM-judge grade the result against the ground truth,
and records everything as a LangSmith EXPERIMENT you can compare over time.

    # from the backend/ directory, AFTER uploading the dataset:
    python -m evals.upload_dataset      # once
    python -m evals.extraction_eval     # each time you want to grade quality

Only IMPORTS from the app (to call the agent) — it never modifies the pipeline.
"""

import asyncio

from dotenv import load_dotenv
from langsmith import aevaluate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.services.extraction_agent import extract_features_from_chunks

load_dotenv()  # LANGSMITH_* (for the experiment) + ANTHROPIC_API_KEY (for the judge)

DATASET_NAME = "extraction-eval"


# --------------------------------------------------------------------------- #
# 1. Target — run the real extraction agent on one chunk                      #
# --------------------------------------------------------------------------- #
# LangSmith calls this with the example's `inputs`; whatever it returns becomes
# the "actual output" the judge sees. (No dedupe here — this evals the agent.)

async def run_extraction(inputs: dict) -> dict:
    features = await extract_features_from_chunks([inputs["chunk_text"]])
    return {"features": features}


# --------------------------------------------------------------------------- #
# 2. LLM-judge — grade actual vs expected (fuzzy, by meaning)                  #
# --------------------------------------------------------------------------- #

class JudgeVerdict(BaseModel):
    score: float = Field(
        description="Overall extraction quality from 0.0 (poor) to 1.0 (perfect): "
        "how many expected features were captured, minus any junk extracted."
    )
    matched: list[str] = Field(
        default_factory=list,
        description="Expected features that WERE captured (matched by meaning).",
    )
    missed: list[str] = Field(
        default_factory=list,
        description="Expected features that were NOT captured.",
    )
    junk: list[str] = Field(
        default_factory=list,
        description="Extracted items that should NOT be features (NFRs, objectives, scaffolding).",
    )
    reasoning: str = Field(description="Brief explanation of the score.")


_judge_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=settings.ANTHROPIC_API_KEY,
).with_structured_output(JudgeVerdict)


JUDGE_SYSTEM_PROMPT = """You are grading a feature-extraction system. You are given the source document chunk, the EXPECTED features a correct extraction should produce (ground truth), guidance on what should NOT be extracted, and the ACTUAL features the system extracted.

Judge by MEANING, not exact wording — an expected feature counts as captured if the actual list contains a feature describing the same capability, however it is named or phrased.

Assess two things:
1. Completeness — which expected features were captured (matched) and which were missed?
2. Junk — did the actual list include anything that should NOT be a feature per the guidance (non-functional/quality requirements, high-level objectives, or document scaffolding)?

Give an overall `score` from 0.0 to 1.0: 1.0 means every expected feature captured and zero junk; deduct for each missed expected feature and for each junk item. Be fair but strict."""


async def extraction_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    chunk = inputs["chunk_text"]
    actual = outputs.get("features", [])
    expected = reference_outputs.get("expected_features", [])
    should_not = reference_outputs.get("should_not_extract", [])

    actual_text = "\n".join(
        f"- {f.get('name', '')}: {f.get('description', '')}" for f in actual
    ) or "(none)"
    expected_text = "\n".join(f"- {e}" for e in expected) or "(none)"
    should_not_text = "\n".join(f"- {s}" for s in should_not) or "(none)"

    user = (
        f"SOURCE CHUNK:\n{chunk}\n\n"
        f"EXPECTED FEATURES (ground truth):\n{expected_text}\n\n"
        f"SHOULD NOT BE EXTRACTED:\n{should_not_text}\n\n"
        f"ACTUAL EXTRACTED FEATURES:\n{actual_text}"
    )

    verdict: JudgeVerdict = await _judge_llm.ainvoke(
        [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user)]
    )

    comment = (
        f"matched {len(verdict.matched)}/{len(expected)} | "
        f"missed={verdict.missed} | junk={verdict.junk} :: {verdict.reasoning}"
    )
    return {"key": "extraction_quality", "score": verdict.score, "comment": comment}


# --------------------------------------------------------------------------- #
# 3. Tie it together — aevaluate creates the Experiment                        #
# --------------------------------------------------------------------------- #

async def main() -> None:
    await aevaluate(
        run_extraction,
        data=DATASET_NAME,
        evaluators=[extraction_quality],
        experiment_prefix="extraction",
        max_concurrency=2,
    )
    print(
        "Eval complete. Open LangSmith -> Datasets & Experiments -> "
        f"'{DATASET_NAME}' -> the newest 'extraction-...' experiment to see the scores."
    )


if __name__ == "__main__":
    asyncio.run(main())
