"""
Push the classify-eval CASES to a LangSmith Dataset (requirements.md §7.2) — "Push #1".

Uploads each case's input (requirement + hardcoded selected_features) and reference
output (expected_status). Records NO results — classify_eval.py produces those later
as Experiments.

Run once, or whenever you edit classify_cases.py. Re-running REPLACES the dataset
so examples never duplicate.

    # from the backend/ directory:
    python -m evals.upload_classify_dataset

Standalone: imports only the CASES + langsmith. It never imports or touches the
matching pipeline.
"""

from dotenv import load_dotenv
from langsmith import Client

from evals.classify_cases import CASES

load_dotenv()  # picks up LANGSMITH_API_KEY etc. from backend/.env

DATASET_NAME = "classify-eval"


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
        description="Classify-node eval: requirement + selected feature set -> expected exact_match/needs_modification.",
    )

    examples = [
        {
            "inputs": {
                "requirement": c["requirement"],
                "selected_features": c["selected_features"],
            },
            "outputs": {
                "expected_status": c["expected_status"],
            },
            "metadata": {"case": c["name"]},
        }
        for c in CASES
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"Uploaded {len(examples)} example(s) to dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
