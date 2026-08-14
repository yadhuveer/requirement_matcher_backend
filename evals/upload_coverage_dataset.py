"""
Push the coverage-eval CASES to a LangSmith Dataset (requirements.md §7.2) — "Push #1".

Uploads each case's input (requirement + hardcoded candidates) and reference
output (expected_relevant + expected_selected). Records NO results — the eval
runner (coverage_eval.py) produces those later as Experiments.

Run once, or whenever you edit coverage_cases.py. Re-running REPLACES the
dataset so examples never duplicate.

    # from the backend/ directory:
    python -m evals.upload_coverage_dataset

Standalone: imports only the CASES + langsmith. It never imports or touches the
matching pipeline.
"""

from dotenv import load_dotenv
from langsmith import Client

from evals.coverage_cases import CASES

load_dotenv()  # picks up LANGSMITH_API_KEY etc. from backend/.env

DATASET_NAME = "coverage-eval"


def main() -> None:
    client = Client()

    # Replace any existing dataset of this name so re-uploads stay clean.
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        existing = None
    if existing is not None:
        client.delete_dataset(dataset_id=existing.id)
        print(f"Existing '{DATASET_NAME}' deleted — re-uploading fresh.")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Coverage-node eval: requirement + hardcoded candidates -> expected covering set.",
    )

    examples = [
        {
            "inputs": {
                "requirement": c["requirement"],
                "candidates": c["candidates"],
            },
            "outputs": {
                "expected_relevant": c["expected_relevant"],
                "expected_selected": c["expected_selected"],
            },
            "metadata": {"case": c["name"]},
        }
        for c in CASES
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"Uploaded {len(examples)} example(s) to dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
