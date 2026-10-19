"""Download the NIST AI Risk Management Framework 1.0 (NIST AI 100-1) into data/sample/.

The PDF is a third-party document: it is NOT part of this repository (it is git-ignored).
Source: https://www.nist.gov/itl/ai-risk-management-framework
NIST states that works by its own employees are not subject to US copyright (17 U.S.C. 105) and asks for
credit; this may not hold outside the US. Check NIST's own notice before reusing the document.

Usage:  python scripts/download_pdf.py [--force]
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
DEST = Path("data/sample/NIST.AI.100-1.pdf")
MAX_BYTES = 100 * 1024 * 1024


def download(url: str, dest: Path, force: bool = False) -> int:
    if dest.exists() and not force:
        print(f"Already present: {dest} (use --force to download again)")
        return 0
    req = urllib.request.Request(url, headers={"User-Agent": "rag-assistant-download/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    if len(data) > MAX_BYTES:
        print("Download failed: file is larger than 100 MB", file=sys.stderr)
        return 1
    if not data.startswith(b"%PDF"):
        print("Download failed: the response is not a PDF", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"Saved {dest} ({len(data) / 1024 / 1024:.1f} MB). Credit: NIST AI 100-1, https://www.nist.gov/itl/ai-risk-management-framework")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args(argv)
    return download(URL, DEST, args.force)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
