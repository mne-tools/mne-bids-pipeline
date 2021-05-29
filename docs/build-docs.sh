#!/bin/bash

set -e
export MKDOCS=1

python docs/source/examples/gen_examples.py

cd docs
mkdocs build
cd ..
