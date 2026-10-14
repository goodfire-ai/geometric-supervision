#!/usr/bin/env python3
"""Check that the environment can run the pipeline: packages, model access, GPU."""
from __future__ import annotations

import json

from huggingface_hub import snapshot_download

from common.model import MODEL_NAME, MODEL_REVISION, runtime_fingerprint


def main() -> None:
    print(json.dumps(runtime_fingerprint(), indent=1))
    try:
        path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
        print(f"model snapshot found in the local Hugging Face cache: {path}")
    except Exception:
        print(f"{MODEL_NAME}@{MODEL_REVISION[:8]} is not in the local Hugging Face cache; it will be "
              "downloaded on first use (requires accepting the Llama 3.1 license on the Hub and `hf auth login`).")
    if not runtime_fingerprint()["gpu"]:
        print("no GPU visible (fine for configs, tests and figures; training needs one GPU)")


if __name__ == "__main__":
    main()
