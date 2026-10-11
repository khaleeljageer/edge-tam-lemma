#!/usr/bin/env bash
# Print the hardware/software facts that bench/devices.md asks for.
set -u
echo "## $(hostname)"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core|max MHz|Architecture"
grep -m1 -E "Model|Hardware" /proc/cpuinfo 2>/dev/null
free -m 2>/dev/null | head -2
grep PRETTY_NAME /etc/os-release 2>/dev/null; uname -srm
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
command -v vcgencmd >/dev/null && { vcgencmd measure_temp; vcgencmd get_throttled; }
python3 --version
python3 - <<'PY' 2>/dev/null
from importlib.metadata import version
for p in ("numpy", "onnxruntime", "ncnn", "ai-edge-litert", "executorch", "torch"):
    try:
        print(f"{p} {version(p)}")
    except Exception:
        pass
PY
