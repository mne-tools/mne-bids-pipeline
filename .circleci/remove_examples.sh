#!/bin/bash

set -eo pipefail

VER=$1
if [ -z "$VER" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi
ROOT="$PWD/$VER/examples/"
if [ ! -d ${ROOT} ]; then
  echo "Version directory does not exist or appears incorrect:"
  echo
  echo "$ROOT"
  echo
  echo "Are you on the gh-pages branch and is the ds000117 directory present?"
  exit 1
fi
if [ ! -d ${ROOT}ds000117 ]; then
  echo "Directory does not exist:"
  echo
  echo "${ROOT}ds000117"
  echo
  echo "Assuming already pruned and exiting."
  exit 0
fi
echo "Pruning examples in ${ROOT} ..."

find $ROOT -type d -name "*" | tail -n +2 | xargs rm -Rf
find $ROOT -name "*.html" -exec sed -i /^\<h2\ id=\"generated-output\"\>Generated/,/^\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $/{d} {} \;
find $ROOT -name "*.html" -exec sed -i '/^  <a href="#generated-output"/,/^  <\/a>$/{d}' {} \;

echo "Done"
