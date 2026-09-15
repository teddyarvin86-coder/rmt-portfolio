#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push this repository to GitHub.
#
# Usage (do NOT hard-code the token into a tracked file):
#
#     GITHUB_TOKEN=ghp_xxxxxxxxxxxx \
#     REPO_URL=https://github.com/<username>/<repo>.git \
#     bash scripts/push_to_github.sh
#
# If the remote repository does not exist yet, create it first at
# https://github.com/new (leave it empty — do NOT add a README or .gitignore,
# the history is already here).
# ---------------------------------------------------------------------------
set -euo pipefail

: "${REPO_URL:?set REPO_URL, e.g. https://github.com/you/rmt-portfolio.git}"
TOKEN="${GITHUB_TOKEN:-}"

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  echo "error: not a git repository" >&2
  exit 1
fi

# Build the authenticated URL without echoing the token.
if [ -n "$TOKEN" ]; then
  PROTO="$(printf '%s' "$REPO_URL" | sed -E 's#^([a-z]+)://.*#\1#')"
  REST="$(printf '%s' "$REPO_URL" | sed -E 's#^[a-z]+://##')"
  PUSH_URL="${PROTO}://x-access-token:${TOKEN}@${REST}"
else
  echo "warning: GITHUB_TOKEN is empty; relying on cached git credentials" >&2
  PUSH_URL="$REPO_URL"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo "pushing main -> $REPO_URL"
git push -u "$PUSH_URL" main

# Always leave the stored remote free of credentials.
git remote set-url origin "$REPO_URL"
echo "done. remote 'origin' points at $REPO_URL (no token stored)."
