#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import urllib.request
import urllib.error
import argparse
from pathlib import Path


def http_get_json(url: str, token: str | None = None):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "lattice-being-importer/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def list_repos(user: str, token: str | None = None) -> list[dict]:
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{user}/repos?per_page=100&page={page}"
        chunk = http_get_json(url, token)
        if not chunk:
            break
        repos.extend(chunk)
        page += 1
    return repos


def list_branches(owner: str, repo: str, token: str | None = None) -> list[dict]:
    branches = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches?per_page=100&page={page}"
        chunk = http_get_json(url, token)
        if not chunk:
            break
        branches.extend(chunk)
        page += 1
    return branches


def run(cmd: list[str], cwd: str | None = None):
    subprocess.run(cmd, cwd=cwd, check=True)


def safe_copytree(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def safe_copyfile(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def import_repo(owner: str, repo: str, work_root: Path, token: str | None = None) -> dict:
    imports_root = work_root / "_imports"
    imports_root.mkdir(parents=True, exist_ok=True)
    clone_dir = imports_root / repo
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    run(["git", "clone", "--depth", "1", f"https://github.com/{owner}/{repo}.git", str(clone_dir)])

    report: dict = {"repo": repo, "copied": []}
    external_root = work_root / "external" / repo
    workflows_src = clone_dir / ".github" / "workflows"
    workflows_dst = work_root / ".github" / "workflows"
    workflows_dst.mkdir(parents=True, exist_ok=True)
    if workflows_src.exists():
        for f in workflows_src.glob("*.*yml") | workflows_src.glob("*.*yaml"):
            # prefix to avoid collisions
            prefixed = workflows_dst / f"{repo}__{f.name}"
            shutil.copy2(f, prefixed)
            report["copied"].append({"type": "workflow", "to": str(prefixed.relative_to(work_root))})

    # common pipeline/tooling locations
    for rel in [
        "infra",
        "pipelines",
        "scripts",
        "services",
        "libs",
        "docker",
        ".devcontainer",
    ]:
        src = clone_dir / rel
        if src.exists():
            dst = external_root / rel
            safe_copytree(src, dst)
            report["copied"].append({"type": "dir", "from": rel, "to": str(dst.relative_to(work_root))})

    # single files of interest
    for fname in [
        "docker-compose.yml",
        "Makefile",
        "compose.yaml",
        ".gitignore",
        "README.md",
    ]:
        srcf = clone_dir / fname
        if srcf.exists():
            dstf = external_root / fname
            safe_copyfile(srcf, dstf)
            report["copied"].append({"type": "file", "from": fname, "to": str(dstf.relative_to(work_root))})

    return report


def main():
    parser = argparse.ArgumentParser(description="Import multi-branch GitHub repos into monorepo")
    parser.add_argument("--user", required=True, help="GitHub user/org")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub token (optional)")
    args = parser.parse_args()

    work_root = Path(__file__).resolve().parents[1]
    owner = args.user
    token = args.token or None

    try:
        repos = list_repos(owner, token)
    except urllib.error.HTTPError as e:
        print(f"Error listing repos: {e}", file=sys.stderr)
        sys.exit(1)

    selected = []
    for r in repos:
        name = r.get("name")
        try:
            branches = list_branches(owner, name, token)
        except urllib.error.HTTPError:
            continue
        if len(branches) > 1:
            selected.append(name)

    inventory = {"owner": owner, "selected": selected, "reports": []}

    for name in selected:
        try:
            rep = import_repo(owner, name, work_root, token)
            inventory["reports"].append(rep)
        except Exception as e:
            inventory["reports"].append({"repo": name, "error": str(e)})

    external_dir = work_root / "external"
    external_dir.mkdir(exist_ok=True)
    inv_path = external_dir / "INVENTORY.json"
    inv_path.write_text(json.dumps(inventory, indent=2))
    print(f"Imported {len(inventory['reports'])} repos. Inventory: {inv_path}")


if __name__ == "__main__":
    main()

