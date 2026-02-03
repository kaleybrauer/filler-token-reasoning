## `scripts/fetch_compose_facts_sources.py`

#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import urllib.request

FILES = {
    "age_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/age_facts.json",
    "atomic_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/atomic_facts.json",
    "static_facts.json": "https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/static_facts.json",
}

def download(url: str, outpath: pathlib.Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    outpath.write_bytes(data)

def main() -> None:
    ap = argparse.ArgumentParser()
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

