#!/bin/bash -e

STEP_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Generating example templates …"
python $STEP_DIR/source/examples/gen_examples.py

echo "Generating pipeline table …"
python $STEP_DIR/source/features/gen_steps.py

echo "Building the documentation …"
cd $STEP_DIR
mkdocs build
