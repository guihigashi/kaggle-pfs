#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" &>/dev/null && pwd)"

MLRUNS_DIR="$DIR/mlruns"

[ ! -d "$MLRUNS_DIR" ] && mkdir -p "$MLRUNS_DIR"

mlflow server \
  --backend-store-uri "sqlite:///$MLRUNS_DIR/mlruns.db" \
  --default-artifact-root "$MLRUNS_DIR" \
  --workers 1

exit 0
