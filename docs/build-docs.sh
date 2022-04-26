#!/bin/bash

set -e
export MKDOCS=1
export MNE_BIDS_STUDY_SCRIPT_PATH=just/a_dummy.py

echo "Generating example templates …"
python docs/source/examples/gen_examples.py


echo "Bulding the documentation …"
cd docs
PYTHONPATH=../ mkdocs build  # ensure ../config.py can be found
cd ..
