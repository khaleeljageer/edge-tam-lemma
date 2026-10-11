#!/usr/bin/env bash
# Pack everything a device needs to run the benchmark (no torch, no training code
# needed) into bench/tmp/device_bundle.tar.gz.
#
#   bash bench/make_device_bundle.sh                 # all runs with exports
#   bash bench/make_device_bundle.sh edit_1M_s13 lstm_1M_s13   # only these runs
#   scp bench/tmp/device_bundle.tar.gz pi@rpi:~/
#   ssh pi@rpi 'mkdir -p tel && tar xzf device_bundle.tar.gz -C tel && cd tel && bash bench/device_run.sh rpi-cm5'
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench/tmp
# Optional arguments: run names to include (default: every run with exports).
roots=$( [ $# -gt 0 ] && printf 'runs/%s ' "$@" || echo runs )
exports=$(find $roots -path '*/export/onnx' -o -path '*/export/ncnn' -o -path '*/export/ncnn-fp16' -o -path '*/export/tflite' -o -path '*/export/pte' | sort)
[ -n "$exports" ] || { echo "no exports under runs/; run tel.export first" >&2; exit 1; }
# Record the code version: commit, plus "-dirty" if there are uncommitted changes.
sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD -- src bench 2>/dev/null || sha="${sha}-dirty"
echo "$sha" > bench/BUNDLE_VERSION
tar czf bench/tmp/device_bundle.tar.gz \
    --exclude='__pycache__' \
    src/tel bench/bench_runner.py bench/device_run.sh bench/device_info.sh bench/BUNDLE_VERSION \
    data/processed/test.tsv data/processed/splits/train.txt \
    $exports
rm bench/BUNDLE_VERSION
echo "wrote bench/tmp/device_bundle.tar.gz (code $sha, $(du -h bench/tmp/device_bundle.tar.gz | cut -f1))"
echo "$exports" | sed 's/^/  /'
