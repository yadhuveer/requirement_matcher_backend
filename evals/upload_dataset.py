"""
Push the extraction-eval CASES to a LangSmith Dataset (requirements.md §7.2) — "Push #1".

This is the ANSWER KEY: it uploads each case's input (chunk_text) and reference
output (expected_features + should_not_extract). It records NO results — the
eval runner (extraction_eval.py) produces those later as Experiments.

Run once, or whenever you edit extraction_cases.py. Re-running REPLACES the
dataset with a fresh copy, so examples never duplicate.

    # from the backend/ directory:
    python -m evals.upload_dataset

Standalone: imports only the CASES + langsmith. It does NOT import or touch the
extraction pipeline.
"""

from dotenv import load_dotenv
from langsmith import Client

from evals.extraction_cases import CASES

load_dotenv()  # picks up LANGSMITH_API_KEY etc. from backend/.env

DATASET_NAME = "extraction-eval"


def main() -> None:
    client = Client()

    # Replace any existing dataset of this name so re-uploads stay clean
    # (avoids duplicate examples piling up).
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        existing = None
    if existing is not None:
        client.delete_dataset(dataset_id=existing.id)
        print(f"Existing '{DATASET_NAME}' deleted — re-uploading fresh.")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Extraction-agent eval: BRD chunk -> expected features (ground truth).",
    )

    examples = [
        {
            "inputs": {"chunk_text": c["chunk_text"]},
            "outputs": {
                "expected_features": c["expected_features"],
                "should_not_extract": c.get("should_not_extract", []),
            },
            "metadata": {"case": c["name"]},
        }
        for c in CASES
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"Uploaded {len(examples)} example(s) to dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
