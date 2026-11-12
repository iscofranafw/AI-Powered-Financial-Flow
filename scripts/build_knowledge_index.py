"""Build per-business-group FAISS knowledge indexes under knowledge_store/.

Usage:
    python scripts/build_knowledge_index.py
    python scripts/build_knowledge_index.py --group bakery
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai.knowledge_rag import CANONICAL_BUSINESS_GROUPS, build_knowledge_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build FAISS IndexFlatIP knowledge stores for UMKM business guidance.'
    )
    parser.add_argument(
        '--group',
        choices=CANONICAL_BUSINESS_GROUPS,
        default=None,
        help='Optional single canonical business group to rebuild.',
    )
    args = parser.parse_args()

    results = build_knowledge_index(jenis_usaha=args.group)
    total_chunks = sum(item['chunk_count'] for item in results.values())
    print(f'Knowledge FAISS build complete ({len(results)} groups, {total_chunks} chunks):')
    for group, info in results.items():
        print(
            f"  - {group}: {info['chunk_count']} chunks -> {info['index_path']}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
