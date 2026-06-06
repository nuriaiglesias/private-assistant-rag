from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.ingestion.reset_vector_store import reset_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the Qdrant vector store collection")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed with deleting the configured Qdrant collection",
    )
    args = parser.parse_args()

    if not args.force:
        raise SystemExit("Use --force to confirm deletion of the Qdrant collection.")

    config = load_config()
    deleted = reset_vector_store(config)
    if deleted:
        print(f"Deleted collection: {config.qdrant_collection}")
    else:
        print(f"Collection not found: {config.qdrant_collection}")


if __name__ == "__main__":
    main()
