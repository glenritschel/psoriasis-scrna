#!/bin/bash
set -e
echo "Creating psoriasis-scrna repo structure..."
mkdir -p data/raw/GSE173706
mkdir -p data/processed
mkdir -p results/tables
mkdir -p figures
mkdir -p logs
touch data/raw/GSE173706/.gitkeep
touch data/processed/.gitkeep
touch results/tables/.gitkeep
touch figures/.gitkeep
touch logs/.gitkeep
echo "Done."
find . -not -path './.git/*' -not -name '*.py' -not -name '*.ipynb' \
       -not -name '*.md' -not -name '*.sh' -not -name '*.yml' | sort
