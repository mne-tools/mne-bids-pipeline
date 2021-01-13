#!/bin/bash

set -e

python docs/source/examples/gen_examples.py

cd docs
mkdocs build
cd ..
