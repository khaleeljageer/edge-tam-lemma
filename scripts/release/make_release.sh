#!/usr/bin/env bash
# Pack the trained models and lookup tables for a GitHub release (models are not in git).
#
#   bash scripts/release/make_release.sh [tag]        # default tag: v1.0
#
# Writes to bench/tmp/release/<tag>/:
#   tel-models-<tag>.tar.gz   the two deployed students (edit_kd_500k_s13, lstm_kd_500k_s13:
#                             PyTorch checkpoint, config, tokenizer, metrics and every export:
#                             ONNX fp32/int8, LiteRT fp32/int8, ncnn) and the teacher
#                             (teacher_lstm), as runs/<name>/...
#   tel-tables-<tag>.tar.gz   hybrid lookup tables, data/interim/hybrid/table_<N>k[_reviewed].tsv
#   wiki_freq-<tag>.tsv.gz    Tamil Wikipedia word-frequency list (tawiki-20261001)
#   SHA256SUMS
# Unpack the tarballs in the repository root; paths match the code (runs/, data/interim/).
# Archives are deterministic (sorted names, fixed mtime and owner).
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=${1:-v1.0}
out=bench/tmp/release/$tag
mkdir -p "$out"
runs=(runs/edit_kd_500k_s13 runs/lstm_kd_500k_s13 runs/teacher_lstm)
for r in "${runs[@]}"; do
    [ -f "$r/model.pt" ] || { echo "missing $r/model.pt" >&2; exit 1; }
done
pack() {  # pack <archive> <paths...>
    local archive=$1; shift
    tar --sort=name --mtime='2026-10-14 00:00:00Z' --owner=0 --group=0 --numeric-owner \
        --exclude='__pycache__' --exclude='*.log' -cf - "$@" | gzip -n -9 > "$out/$archive"
}
pack "tel-models-$tag.tar.gz" "${runs[@]}"
pack "tel-tables-$tag.tar.gz" data/interim/hybrid/table_*k.tsv data/interim/hybrid/table_*k_reviewed.tsv
gzip -n -9 -c data/interim/wiki_freq.tsv > "$out/wiki_freq-$tag.tsv.gz"
(cd "$out" && sha256sum ./*.gz > SHA256SUMS)
ls -lh "$out"
cat "$out/SHA256SUMS"
