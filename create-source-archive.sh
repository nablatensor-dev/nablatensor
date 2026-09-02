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

file_list="$(mktemp)"
trap 'rm -f "$file_list"' EXIT

cd "$ROOT"
find . \
  \( -type d \( \
    -name .git -o -name .idea -o -name .vscode -o \
    -name target -o -name build -o -name out -o -name dist -o \
    -name node_modules -o -name .venv -o -name venv -o \
    -name __pycache__ -o -name .ipynb_checkpoints -o \
    -name '*.egg-info' \
  \) -prune \) -o \
  \( -type f \( \
    -name '*.java' -o -name '*.py' -o -name '*.sh' -o \
    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o \
    -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o \
    -name '*.cl' -o -name '*.glsl' -o -name '*.comp' -o \
    -name '*.md' -o -name '*.txt' -o -name '*.adoc' -o -name '*.rst' -o \
    -name '*.xml' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o \
    -name '*.json' -o -name '*.properties' -o -name '*.ipynb' -o \
    -name '.gitignore' -o -name '.gitattributes' -o \
    -name 'LICENSE' -o -name 'NOTICE' -o -name 'Makefile' \
  \) -print0 \) | sort -z > "$file_list"

[[ -s "$file_list" ]] || { echo "no source files found" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

tar --null --files-from="$file_list" \
  --transform="s,^\./,$PROJECT/," \
  --create --gzip --file="$OUTPUT"

printf 'Created %s (%s files)\n' "$OUTPUT" "$(tr -cd '\0' < "$file_list" | wc -c)"