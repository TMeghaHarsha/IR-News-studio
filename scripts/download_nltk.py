"""Download required NLTK resources for the IR project.

Usage:
    python scripts\download_nltk.py

This is an alternative to running `python -c "import nltk; nltk.download(... )"`.
Note: NLTK corpus/data are not Python packages and therefore do NOT belong in
`requirements.txt`. This script simply automates the `nltk.download(...)` calls.
"""
import sys
import nltk


def main() -> int:
    resources = [
        "stopwords",
        "punkt",
        "wordnet",
        "omw-1.4",
    ]

    # Try to find and only download missing resources
    missing = []
    for r in resources:
        try:
            # common resource locations
            if r in ("punkt",):
                nltk.data.find(f"tokenizers/{r}")
            else:
                nltk.data.find(f"corpora/{r}")
        except LookupError:
            missing.append(r)

    if not missing:
        print("All required NLTK resources are already installed.")
        return 0

    for r in missing:
        print(f"Downloading NLTK resource: {r}")
        nltk.download(r)

    print("NLTK setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
