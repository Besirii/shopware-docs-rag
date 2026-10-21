"""Fetch the official Shopware docs at a pinned commit.

Design notes
------------
* Shallow (`--depth 1`) + partial (`--filter=blob:none`) + sparse clone: we only
  download the blobs of the folders we index (plus `snippets/`, which pages
  include). Images under `assets/` are never fetched.
* `core.longpaths=true`: some doc paths exceed Windows' 260-char MAX_PATH.
* The commit SHA is written to `manifest.json`. Every chunk and every eval result
  carries it, so a number in the README can always be traced to an exact docs
  version.
* The docs repository has no license file, so this project never redistributes
  the content - only this script that downloads it.

Usage:  python -m ingest.fetch_docs [--commit SHA] [--ref main]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# Pages inline content from here via `<!--@include: ...-->` and `<<< @/docs/snippets/...`.
SUPPORT_DIRS = ["snippets"]
MANIFEST_NAME = "manifest.json"


def _git(repo: Path | None, *args: str, capture: bool = False) -> str:
    cmd = ["git", "-c", "core.longpaths=true"]
    if repo is not None:
        cmd += ["-C", str(repo)]
    cmd += list(args)
    log.debug("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip() if capture else ""


def fetch_docs(settings: Settings, commit: str | None = None, ref: str | None = None) -> dict:
    repo = settings.docs_dir
    commit = commit or settings.docs_commit
    ref = ref or settings.docs_ref
    target = commit or ref

    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning %s into %s", settings.docs_repo_url, repo)
        _git(None, "init", "--quiet", str(repo))
        _git(repo, "config", "core.longpaths", "true")
        _git(repo, "remote", "add", "origin", settings.docs_repo_url)

    # Fetching a specific SHA works on GitHub (allowReachableSHA1InWant), so the
    # same code path handles "latest main" and "pinned commit".
    log.info("Fetching %s (depth 1, no blobs yet)", target)
    _git(repo, "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", target)

    sparse_dirs = list(dict.fromkeys([*settings.docs_include, *SUPPORT_DIRS]))
    # Cone mode takes plain directory names - no leading slash (Git Bash would
    # rewrite "/guides" into a Windows path).
    _git(repo, "sparse-checkout", "set", "--cone", *sparse_dirs)
    # Checkout downloads exactly the blobs of the sparse folders, in one batch.
    _git(repo, "checkout", "--quiet", "--force", "--detach", "FETCH_HEAD")

    sha = _git(repo, "rev-parse", "HEAD", capture=True)
    committed_at = _git(repo, "show", "-s", "--format=%cI", "HEAD", capture=True)
    manifest = {
        "repo_url": settings.docs_repo_url,
        "ref": ref,
        "commit_sha": sha,
        "committed_at": committed_at,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sparse_dirs": sparse_dirs,
    }
    (repo.parent / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Docs at commit %s (%s)", sha[:12], committed_at)
    return manifest


def read_manifest(settings: Settings) -> dict:
    path = settings.docs_dir.parent / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m ingest.fetch_docs` first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", help="exact commit SHA to pin (overrides DOCS_COMMIT)")
    parser.add_argument("--ref", help="branch or tag (overrides DOCS_REF)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = fetch_docs(get_settings(), commit=args.commit, ref=args.ref)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
