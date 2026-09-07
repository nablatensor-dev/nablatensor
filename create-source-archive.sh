#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(basename "$ROOT")"
OUTPUT="${1:-$ROOT/${PROJECT}-source-$(date +%Y%m%d).tar.gz}"

case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$PWD/$OUTPUT" ;;
esac

command -v tar >/dev/null || { echo "tar is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }

file_list="$(mktemp)"
trap 'rm -f "$file_list"' EXIT

cd "$ROOT"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }

# The index, not the working tree: the archive holds what is tracked.
git ls-files -z --cached --exclude-standard > "$file_list"

[[ -s "$file_list" ]] || { echo "no tracked files found" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

tar --null --files-from="$file_list" \
  --transform="s,^,$PROJECT/," \
  --create --gzip --file="$OUTPUT"

printf 'Created %s (%s files)\n' "$OUTPUT" "$(tr -cd '\0' < "$file_list" | wc -c)"