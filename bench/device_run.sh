#!/usr/bin/env bash
# Run on the device, from the unpacked bundle:  bash bench/device_run.sh <device-name> [threads...]
# Installs onnxruntime + ncnn + ai-edge-litert in a local venv (no torch), records device
# info, and benchmarks every export whose runtime imports. ExecuTorch exports need
# torch on the device; they are included only if executorch is importable.
# If the bundle has hybrid lookup tables (data/interim/hybrid/table_<N>k.tsv), a second
# pass benchmarks model + table on a frequency-weighted token stream (tel.hybrid).
# HYBRID_ONLY=1 bash bench/device_run.sh <device>   skips the plain pass;
# NO_HYBRID=1 skips the hybrid pass.
# FORMATS="tflite tflite-int8dyn" bash bench/device_run.sh <device>   only those exports.
set -euo pipefail
device=${1:?usage: device_run.sh <device-name> [threads...]}
shift
threads=${*:-"1 $(nproc)"}
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
# Always (re)install: a .venv left over from an older bundle may lack new runtimes.
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy onnxruntime ncnn ai-edge-litert pyyaml || true

# Export dir suffix -> Python module of its runtime.
declare -A module=([onnx]=onnxruntime [onnx-int8]=onnxruntime [onnx-int8dyn]=onnxruntime
                  [ncnn]=ncnn [ncnn-fp16]=ncnn [tflite]=ai_edge_litert
                  [tflite-int8dyn]=ai_edge_litert [pte]=executorch)
models=""
for suffix in ${FORMATS:-onnx onnx-int8 onnx-int8dyn ncnn ncnn-fp16 tflite tflite-int8dyn pte}; do
    [ -n "${module[$suffix]:-}" ] || { echo "unknown export format: $suffix" >&2; exit 1; }
    dirs=$(ls -d runs/*/export/$suffix 2>/dev/null || true)
    [ -n "$dirs" ] || continue
    if .venv/bin/python -c "import ${module[$suffix]}" 2>/dev/null; then
        models="$models $dirs"
    else
        echo "skipping $suffix exports: python module '${module[$suffix]}' not importable" >&2
    fi
done
[ -n "$models" ] || { echo "no runnable exports found" >&2; exit 1; }

governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
[ "$governor" = performance ] || echo "WARNING: CPU governor is '$governor', not 'performance'" >&2

mkdir -p bench/results
bash bench/device_info.sh > "bench/results/${device}.info.txt" 2>&1 || true
files="bench/results/${device}.info.txt"
if [ "${HYBRID_ONLY:-0}" != 1 ]; then
    PYTHONPATH=src .venv/bin/python bench/bench_runner.py --device "$device" \
        --threads $threads --models $models
    files="$files bench/results/${device}.csv"
fi
tables=$(ls data/interim/hybrid/table_*k.tsv 2>/dev/null | sort -V || true)
if [ -n "$tables" ] && [ "${NO_HYBRID:-0}" != 1 ]; then
    PYTHONPATH=src .venv/bin/python bench/bench_runner.py --device "$device" \
        --threads $threads --models $models --weighted --tables none $tables \
        --out "bench/results/${device}-hybrid.csv"
    files="$files bench/results/${device}-hybrid.csv"
fi
echo "copy back: $files"
