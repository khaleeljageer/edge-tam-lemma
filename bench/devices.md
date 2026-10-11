# Benchmark devices

Record one section per device. Fill every field before running; re-check the
governor before each session. `bench/device_info.sh` prints most of these.

## desktop

| Field | Value |
|---|---|
| Role | x86 reference |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads, max 5.66 GHz |
| RAM | 32 GB |
| OS / kernel | Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic |
| Governor / power mode | `powersave` (amd-pstate EPP default) |
| Python | 3.11 (uv) |
| Runtimes | onnxruntime 1.31.0, ncnn 1.0.20260526, ai-edge-litert 2.2.0, executorch 1.5.1 |
| Threads tested | 1, 24 |

## rpi5

| Field | Value |
|---|---|
| Role | ARM CPU-only edge baseline |
| Board | Raspberry Pi 5 Model B Rev 1.1 (same BCM2712 SoC as the CM5 in the research plan) |
| CPU | 4× Cortex-A76, max 2.4 GHz, no SMT |
| RAM | 8 GB |
| OS / kernel | Debian GNU/Linux 13 (trixie), Linux 6.18.50+rpt-rpi-2712 |
| Governor / power mode | `performance` (first 2026-10-10 run used `ondemand`; timings matched within ~3%) |
| Cooling | Active: Waveshare aluminium alloy case with dual cooling fans (case acts as heatsink); 37.3 °C idle, throttled=0x0 before the run |
| Python | 3.13.5 (system) |
| Runtimes | onnxruntime 1.31.0, ncnn 1.0.20260526, ai-edge-litert 2.3.0 (desktop has 2.2.0) |
| Threads tested | 1, 4 |

## android-phone

_TBD_

## qualcomm-auto-board

_TBD (if feasible)_
