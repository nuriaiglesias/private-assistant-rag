from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.ingestion.ingest import ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--input",
        help="Directory with documents to ingest",
        default="data/raw",
    )
    args = parser.parse_args()

    config = load_config()
    repo_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    data_root = input_path if input_path.is_absolute() else repo_root / input_path
    data_root = data_root.resolve()

    try:
        total = ingest_documents(data_root, config)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"Indexed {total} chunks into {config.qdrant_collection}")


if __name__ == "__main__":
    main()
