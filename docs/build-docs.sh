#!/bin/bash

set -e
export MKDOCS=1
export MNE_BIDS_STUDY_SCRIPT_PATH=just/a_dummy.py

source ~/python-venv/bin/activate
python docs/source/examples/gen_examples.py

cd docs
mkdocs build
cd ..
