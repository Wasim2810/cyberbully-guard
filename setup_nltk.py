"""
setup_nltk.py
-------------
One-time setup script to download the NLTK corpora required by
utils/preprocessing.py (stopwords, WordNet, punkt tokenizer).

Run once after installing requirements:
    python setup_nltk.py

If this step is skipped or fails (e.g. no internet access), the
application will still run correctly using the built-in lightweight
fallback preprocessing implemented in utils/preprocessing.py.
"""

import nltk

PACKAGES = ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]

if __name__ == "__main__":
    for pkg in PACKAGES:
        try:
            print(f"Downloading NLTK package: {pkg} ...")
            nltk.download(pkg)
        except Exception as exc:
            print(f"Could not download '{pkg}': {exc}")
    print("\nNLTK setup complete. You can now run: python train_model.py")
