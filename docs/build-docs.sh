#!/bin/bash -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Generating example templates …"
python $SCRIPT_DIR/source/examples/gen_examples.py

echo "Building the documentation …"
cd $SCRIPT_DIR
PYTHONPATH=../ mkdocs build
