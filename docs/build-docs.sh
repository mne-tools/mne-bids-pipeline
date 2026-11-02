#!/bin/bash

set -eo pipefail

STEP_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Generating example templates …"
python $STEP_DIR/source/examples/gen_examples.py

echo
echo "Generating pipeline table …"
python $STEP_DIR/source/features/gen_steps.py

echo
echo "Generating config docs …"
python $STEP_DIR/source/settings/gen_settings.py

echo
echo "Building the documentation …"
cd $STEP_DIR
mkdocs build --strict
