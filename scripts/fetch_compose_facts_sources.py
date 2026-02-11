#!/usr/bin/env python3
"""
Download compose_facts source JSON files from GitHub.
"""
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

FILES = {
    "age_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/age_facts.json",
    "atomic_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/atomic_facts.json",
    "static_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/static_facts.json",
}

# Download settings
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def download(url: str, outpath: pathlib.Path) -> None:
    """Download a URL to a file with timeout and retries."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as r:
                data = r.read()
            outpath.write_bytes(data)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {MAX_RETRIES} attempts"
                ) from last_error


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download compose_facts source JSON files from GitHub."
    )
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for JSON sources")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        outpath = outdir / fname
        if outpath.exists() and outpath.stat().st_size > 0:
            print(f"[skip] {fname} already exists")
            continue
        print(f"[download] {fname}")
        download(url, outpath)

        # quick sanity parse
        try:
            json.loads(outpath.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Downloaded file is not valid JSON: {outpath}") from e

    print(f"Done. Wrote sources to: {outdir}")


if __name__ == "__main__":
    main()