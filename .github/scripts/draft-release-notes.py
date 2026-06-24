"""Draft release notes for a Strands Agents package via Bedrock.

The maintainer types the explicit version at workflow_dispatch time. This
script only drafts the prose — it does not influence the version in any way.

Inputs (env):
    PACKAGE         "python" or "typescript"
    PREV_TAG        previous release tag (e.g. "python/v1.2.3")
    NEW_REF         git ref or SHA being released (typically a pinned SHA)
    BEDROCK_MODEL   inference profile / model id (defaults to a Claude Sonnet on Bedrock)
    AWS_REGION      defaults to us-west-2 (matches strands-command.yml)

Output (file in $GITHUB_WORKSPACE):
    release-notes.md    markdown release notes

Exit codes:
    0   notes drafted successfully
    1   no commits between PREV_TAG and NEW_REF
    2   Bedrock call failed or returned empty output
        (the caller workflow falls back to `git shortlog` on non-zero exit)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


WORKSPACE = Path(os.environ.get("GITHUB_WORKSPACE", "."))
NOTES_PATH = WORKSPACE / "release-notes.md"


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def collect_commits(prev_tag: str, new_ref: str) -> str:
    """Return the git log between prev_tag and new_ref, one commit per block."""
    return run(
        [
            "git",
            "log",
            f"{prev_tag}..{new_ref}",
            "--pretty=format:--- COMMIT ---%n%H%n%an%n%s%n%b",
            "--no-merges",
        ]
    )


def collect_diff_stats(prev_tag: str, new_ref: str) -> str:
    """Return per-file change stats so the model can reason about scope."""
    return run(["git", "diff", "--stat", f"{prev_tag}...{new_ref}"])


def build_prompt(
    package: str,
    prev_tag: str,
    new_ref: str,
    commits: str,
    diff_stats: str,
) -> str:
    return f"""You are drafting release notes for the {package} package of the Strands Agents SDK.

Previous tag: `{prev_tag}`
New ref: `{new_ref}`

## Your task

Draft user-facing release notes in markdown, grouped under these headings
(omit headings with no entries):

- **Breaking changes**
- **Features**
- **Bug fixes**
- **Other changes** (docs, refactors, internal)

Each bullet should be one line, written for users, referencing the commit
hash in parens.

## Output format

Return ONLY the markdown release notes. No JSON wrapping, no markdown
fences around the whole response, no prose introduction or conclusion. The
first character of your response should be the first character of the
notes (typically `#` or `*`).

## Commits since last release

```
{commits}
```

## Diff stats

```
{diff_stats}
```
"""


def call_bedrock(prompt: str, model_id: str, region: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def main() -> int:
    package = os.environ["PACKAGE"]
    prev_tag = os.environ["PREV_TAG"]
    new_ref = os.environ["NEW_REF"]
    model_id = os.environ.get(
        "BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    region = os.environ.get("AWS_REGION", "us-west-2")

    commits = collect_commits(prev_tag, new_ref)
    if not commits.strip():
        print(f"No commits between {prev_tag} and {new_ref} — nothing to release.")
        return 1

    diff_stats = collect_diff_stats(prev_tag, new_ref)
    prompt = build_prompt(package, prev_tag, new_ref, commits, diff_stats)

    try:
        notes = call_bedrock(prompt, model_id, region)
    except (BotoCoreError, ClientError) as exc:
        # Network / IAM / throttling. Caller workflow handles fallback.
        print(f"Bedrock call failed: {exc}", file=sys.stderr)
        return 2

    if not notes:
        print("Bedrock returned empty notes.", file=sys.stderr)
        return 2

    NOTES_PATH.write_text(notes)
    print(f"Wrote {NOTES_PATH} ({len(notes)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
