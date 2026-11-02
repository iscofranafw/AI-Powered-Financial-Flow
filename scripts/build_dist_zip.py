#!/usr/bin/env python3
"""Build and verify a clean distribution zip archive for the repository.

Excludes:
- vector_store/ (local FAISS vector indices and metadata)
- data/Daftar_Transaksi_*.csv (real store transaction data)
- data/*.csv (top-level store CSVs, preserving data/samples/*.csv)
- .env* (environment configuration secrets, preserving .env.example)
- __pycache__/ (*.pyc compiled artifacts)
- .git/ (version control metadata)
- .pytest_cache/, .mypy_cache/, .ruff_cache/
- .idea/, .vscode/, .antigravity/
- *.zip archives

Verifies:
- 0 vector_store/ entries
- 0 data/Daftar_Transaksi_*.csv entries
- 0 secret .env entries
- Preserves all data/samples/*.csv benchmark datasets
Fails with exit code 1 if any excluded path is present or verification fails.
"""

import os
import re
import sys
import zipfile
from pathlib import Path


if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Safe fallback: terminal stdout/stderr reconfiguration failure should not prevent packaging
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    'vector_store',
    'vector_indexes',
    '.venv',
    'venv',
    '.idea',
    '.vscode',
    '.antigravity',
}


def should_exclude(rel_path: str) -> bool:
    # Normalize path to POSIX forward slash
    norm = rel_path.replace('\\', '/')
    parts = norm.split('/')

    # Check directory components
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True

    filename = parts[-1]

    # Exclude __pycache__ at any level
    if '__pycache__' in parts:
        return True

    # Exclude zip archives themselves
    if filename.endswith('.zip'):
        return True

    # Exclude vector_store at any level
    if 'vector_store' in parts:
        return True

    # Exclude top-level real store CSVs: data/Daftar_Transaksi_*.csv or data/*.csv
    if norm.startswith('data/') and len(parts) == 2:
        if filename.startswith('Daftar_Transaksi_') and filename.endswith('.csv'):
            return True
        if filename.endswith('.csv') and not norm.startswith('data/samples/'):
            return True

    # Exclude secrets (.env, .env.* but allow .env.example)
    if filename == '.env' or (filename.startswith('.env.') and filename != '.env.example'):
        return True
    if filename == 'secrets.toml':
        return True

    # Exclude compiled python artifacts and temp files
    if filename.endswith(('.pyc', '.pyo', '.pyd', '.swp', '.swo', '.tmp', '.bak')):
        return True

    return False


def build_dist_zip(output_path: Path) -> Path:
    print(f"Building clean distribution zip from: {REPO_ROOT}")
    print(f"Target archive: {output_path}")

    # Remove existing archive if present
    if output_path.exists():
        output_path.unlink()

    included_count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            # Prune excluded directories in-place for fast traversal
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.git')]

            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(REPO_ROOT).as_posix()

                if should_exclude(rel_path):
                    continue

                zf.write(abs_path, arcname=rel_path)
                included_count += 1

    print(f"Packaged {included_count} files into {output_path.name}")
    return output_path


def verify_dist_zip(zip_path: Path) -> bool:
    print(f"\nVerifying distribution archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()

    errors = []

    # 1. Check vector_store exclusion
    vs_entries = [n for n in namelist if 'vector_store' in n.lower()]
    if vs_entries:
        errors.append(f"vector_store entries found ({len(vs_entries)}): {vs_entries[:3]}")

    # 2. Check real transaction CSV exclusion
    dt_entries = [n for n in namelist if re.search(r'data/Daftar_Transaksi_.*\.csv$', n, re.I)]
    if dt_entries:
        errors.append(f"Real transaction CSV entries found ({len(dt_entries)}): {dt_entries}")

    # Also ensure no top-level data/*.csv exist
    top_csv = [n for n in namelist if re.match(r'^data/[^/]+\.csv$', n, re.I)]
    if top_csv:
        errors.append(f"Top-level data/*.csv entries found ({len(top_csv)}): {top_csv}")

    # 3. Check secret .env exclusion
    env_entries = [n for n in namelist if re.search(r'(^|/)\.env($|\..*)', n) and not n.endswith('.env.example')]
    if env_entries:
        errors.append(f"Secret .env entries found ({len(env_entries)}): {env_entries}")

    # 4. Check __pycache__ exclusion
    pycache_entries = [n for n in namelist if '__pycache__' in n or n.endswith('.pyc')]
    if pycache_entries:
        errors.append(f"__pycache__ entries found ({len(pycache_entries)}): {pycache_entries[:3]}")

    # 5. Check .git exclusion
    git_entries = [n for n in namelist if n.startswith('.git/') or '/.git/' in n]
    if git_entries:
        errors.append(f".git entries found ({len(git_entries)}): {git_entries[:3]}")

    # 6. Check that data/samples/*.csv ARE present
    sample_entries = sorted([n for n in namelist if re.match(r'^data/samples/[^/]+\.csv$', n, re.I)])
    if not sample_entries:
        errors.append("data/samples/*.csv files are missing from archive!")

    if errors:
        print("[FAIL] VERIFICATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        return False

    print("[OK] VERIFICATION PASSED:")
    print(f"   * vector_store/ entries: 0")
    print(f"   * data/Daftar_Transaksi_*.csv entries: 0")
    print(f"   * .env* secret entries: 0")
    print(f"   * __pycache__/ entries: 0")
    print(f"   * .git/ entries: 0")
    print(f"   * data/samples/*.csv preserved ({len(sample_entries)} files):")
    for s in sample_entries:
        print(f"     - {s}")
    print(f"   * Total archived files: {len(namelist)}")
    return True


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else 'financial-flow-dist.zip'
    target_path = Path(target_name)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    build_dist_zip(target_path)
    success = verify_dist_zip(target_path)

    if not success:
        sys.exit(1)
    print(f"\nDistribution archive ready: {target_path}")


if __name__ == '__main__':
    main()
