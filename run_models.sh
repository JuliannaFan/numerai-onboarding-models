#!/bin/sh
set -eu
cd "$(dirname "$0")"
# The scikit-learn macOS wheel bundles the OpenMP library LightGBM needs.
if [ "$(uname -s)" = Darwin ]; then
  NUMERAI_OMP_DIR="$PWD/.venv312/lib/python3.12/site-packages/sklearn/.dylibs"
  export DYLD_LIBRARY_PATH="$NUMERAI_OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
exec .venv312/bin/python -u prepare_models.py
