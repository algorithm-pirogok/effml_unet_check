#!/usr/bin/env bash
# Downloads Carvana train.zip + train_masks.zip from Kaggle into ~/effml/data/carvana/.
# Requires: ~/.kaggle/kaggle.json with Carvana competition rules accepted on
# kaggle.com first. Run from ~/effml/.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/effml/data/carvana}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

KAGGLE="${KAGGLE:-$HOME/effml/.venv/bin/kaggle}"

echo "downloading train.zip + train_masks.zip ..."
"$KAGGLE" competitions download -c carvana-image-masking-challenge -f train.zip       --quiet
"$KAGGLE" competitions download -c carvana-image-masking-challenge -f train_masks.zip --quiet

echo "extracting ..."
unzip -q -o train.zip
unzip -q -o train_masks.zip

ls -la "$DATA_DIR" | head -10
echo "train images:  $(ls train       2>/dev/null | wc -l)"
echo "train masks:   $(ls train_masks 2>/dev/null | wc -l)"
