#!/bin/bash

# This script uses 'find' to add files recursively.
# This is the most compatible method and will work in
# older shells or environments where 'globstar' (**) is not available.

echo "Finding and staging all .py, .yaml, .json, .txt, and .sh files..."

find . -type f \( \
  -name "*.py"   -o \
  -name "*.yaml" -o \
  -name "*.json" -o \
  -name "*.jsonl" -o \
  -name "*.html" -o \
  -name "*.toml" -o \
  -name "*.csv" -o \
  -name "*.txt"  -o \
  -name "*.sh" \
\) -exec git add {} +

echo "Done. Run 'git status' to see the changes."
