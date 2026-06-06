from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.ingestion.scraper import DEFAULT_SEEDS_PATH, OUTPUT_DIR, load_seed_urls, scrape


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UNIR public pages from seed URLs")
    parser.add_argument(
        "--seeds",
        default=str(DEFAULT_SEEDS_PATH),
        help="Path to a JSONL file with seed URLs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of seeds to process",
    )
    args = parser.parse_args()

    seed_path = Path(args.seeds)
    seed_urls = load_seed_urls(seed_path)
    if args.limit is not None:
        seed_urls = seed_urls[: args.limit]

    outputs = scrape(seed_urls)
    print(f"Saved {len(outputs)} documents to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
