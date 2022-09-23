#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e
export MKDOCS=1
export MNE_BIDS_STUDY_SCRIPT_PATH=just/a_dummy.py

echo "Generating example templates …"
python $SCRIPT_DIR/source/examples/gen_examples.py

echo "Building the documentation …"
cd $SCRIPT_DIR
PYTHONPATH=../ mkdocs build
