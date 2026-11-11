#!/usr/bin/env bash
# Reproducible end-to-end verification of the alignment core.
#
# Generates synthetic multi-track material with KNOWN fixed delay and sample
# clock drift (plus a reference gap and a member interruption), runs the full
# detect -> fit -> render -> verify pipeline, and fails if the numerical
# tolerances are not met. No network or GUI is required.
set -euo pipefail

BIN="${BIN:-target/debug/align-cli}"
WORK="${WORK:-/tmp/band-align-verify}"
RATE=48000
DURATION=120
DELAY=0.25
DRIFT_PPM=150
CHANNELS=2

# Acceptable estimator errors for this scenario.
MAX_DELAY_ERROR_MS=2.0
MAX_DRIFT_ERROR_PPM=25.0
MAX_RMS_RESIDUAL_MS=2.0

rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

echo "== synthesize (delay=${DELAY}s drift=${DRIFT_PPM}ppm) =="
"$OLDPWD/$BIN" synthesize \
  --out synth --rate "$RATE" --duration "$DURATION" \
  --delay "$DELAY" --drift-ppm "$DRIFT_PPM" --channels "$CHANNELS" \
  --ref-gap 40:45 --member-gap 80:83

echo "== init project (gap metadata included) =="
"$OLDPWD/$BIN" init-project \
  --name verify --reference synth/reference.wav --member synth/member.wav \
  --ref-gap 40:45 --member-gap 80:83 --out project.json

echo "== align =="
"$OLDPWD/$BIN" align --project project.json --out aligned

echo "== export audit positions =="
"$OLDPWD/$BIN" export-positions --project project.json --out positions.csv

echo "== numerical verification =="
"$OLDPWD/$BIN" verify-project \
  --project project.json --truth synth/truth.json \
  --aligned-dir aligned --out verification.json

python3 - "$WORK" "$MAX_DELAY_ERROR_MS" "$MAX_DRIFT_ERROR_PPM" "$MAX_RMS_RESIDUAL_MS" <<'PY'
import json, sys
work, max_delay_ms, max_drift_ppm, max_rms_ms = sys.argv[1:5]
v = json.load(open(f"{work}/verification.json"))
e = v["estimator"]
checks = [
    ("delay error ms", e["delay_error_secs"] * 1000, float(max_delay_ms)),
    ("drift error ppm", abs(e["drift_error_ppm"]), float(max_drift_ppm)),
    ("rms residual ms", e["rms_residual_secs"] * 1000, float(max_rms_ms)),
    ("interchannel max diff", v["channel"]["max_interchannel_diff"], 1e-6),
]
ok = True
for name, got, limit in checks:
    status = "PASS" if got <= limit else "FAIL"
    if got > limit:
        ok = False
    print(f"  [{status}] {name}: {got:.6f} (limit {limit})")
summary = v["summaries"][0]
print(f"  anchors: {summary['anchors_used']} used, "
      f"{summary['anchors_rejected']} rejected, "
      f"{summary['anchors_unknown']} unknown")
if not ok:
    sys.exit("verification failed: a numerical limit was exceeded")
print("all numerical checks passed")
PY

echo "artifacts in $WORK:"
ls -1 "$WORK" "$WORK/aligned"
