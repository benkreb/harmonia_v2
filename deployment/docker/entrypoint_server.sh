#!/usr/bin/env bash
set -euo pipefail
python run_fl.py simulate --config "${FL_CONFIG:-benchmark/configs/f1_iid.yaml}" "${SMOKE_FLAG:---smoke}"
