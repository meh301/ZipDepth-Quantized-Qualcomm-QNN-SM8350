#!/bin/bash
# Compile ZipDepth QDQ variants -> SM8350/V68 HTP context binaries -> EPContext ONNX.
# Sources are COPIED from the (read-only) F:\xreal_depth sweep; all outputs land in
# this scratchpad dir + the standalone repo's assets/models. Nothing on F: is written.
set -u
BUILD="$(cd "$(dirname "$0")" && pwd)"
QAIRT_W='F:\v2.48.0.260626\qairt\2.48.0.260626'
QAIRT_U='/f/v2.48.0.260626/qairt/2.48.0.260626'
QPY='/f/xreal_depth/qairtenv/Scripts/python.exe'
SWEEP='/f/xreal_depth/zipdepth_sweep'
REPO='/c/Users/kenchitaru-alex/Documents/GitHub/zipdepth_NPU'
MODELS_OUT="$REPO/app/src/main/assets/models"
SYSPY='/c/Users/kenchitaru-alex/AppData/Local/Programs/Python/Python314/python.exe'

mkdir -p "$BUILD/src" "$BUILD/dlc" "$BUILD/ctx" "$MODELS_OUT"

# Own copies of the HTP compile config. IMPORTANT: NO soc_id — stamping
# socModel=30 makes QnnContext_createFromBinary REJECT the binary (error 1002)
# on this Samsung SD888, because the unsigned PD cannot query the DSP's SoC id
# to prove a match. The shipped (working) zipdepth binary has socModel=0 with
# dsp_arch v68 only; replicate that.
BUILD_W="$(cygpath -w "$BUILD" | sed 's/\\/\//g')"
cat > "$BUILD/htp_config.json" <<EOF
{
  "devices": [
    {
      "dsp_arch": "v68",
      "cores": [
        { "core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100 }
      ]
    }
  ]
}
EOF
cat > "$BUILD/backend_ext.json" <<EOF
{
  "backend_extensions": {
    "shared_library_path": "QnnHtpNetRunExtensions.dll",
    "config_file_path": "$BUILD_W/htp_config.json"
  }
}
EOF

export PATH="$QAIRT_U/bin/x86_64-windows-msvc:$QAIRT_U/lib/x86_64-windows-msvc:$PATH"
export PYTHONPATH="$QAIRT_W\\lib\\python"
export PYTHONUTF8=1

VARIANTS="a8w8_384 a16w8_320 a8w8_320 a16w8_288 a8w8_288 a8w8_256"
SUMMARY="$BUILD/summary.txt"
: > "$SUMMARY"

for tag in $VARIANTS; do
  t0=$(date +%s)
  # src/ is populated by zd_fullquant.py: FULLY-quantized QDQ (every op type),
  # because any float segment becomes an fp16 HTP op and the V68 skel lacks
  # fp16 -> on-device createFromBinary error 1002. Do NOT use the zipdepth_sweep
  # QDQ files here — those are default-op-set quantizations for CPU eval only.
  src="$BUILD/src/zipdepth_${tag}.onnx"
  [ -f "$src" ] || { echo "$tag: MISSING SOURCE (run zd_fullquant.py first)" >> "$SUMMARY"; continue; }
  dlc="$BUILD/dlc/${tag}.dlc"
  ctxdir="$BUILD/ctx/${tag}"
  mkdir -p "$ctxdir"

  echo "[$(date +%H:%M:%S)] $tag: converter" | tee -a "$SUMMARY"
  # --preserve_io_datatype keeps graph I/O float32 (like the shipped binary) so
  # the float-feeding app works unchanged; the interior is fully fixed-point.
  # Safe ONLY on a fully-quantized QDQ — with float segments in-graph this
  # produced unpreparable fp32 ops (the earlier finalize err 1002).
  "$QPY" "$QAIRT_U/bin/x86_64-windows-msvc/qairt-converter" \
      --input_network "$(cygpath -w "$src")" \
      --preserve_io_datatype \
      --output_path "$(cygpath -w "$dlc")" > "$ctxdir/convert.log" 2>&1
  [ -f "$dlc" ] || { echo "$tag: DLC FAILED (see ctx/$tag/convert.log)" >> "$SUMMARY"; continue; }

  echo "[$(date +%H:%M:%S)] $tag: context-binary-generator" | tee -a "$SUMMARY"
  # --htp_socs sm8350, exactly like the shipped build. Do NOT use the LAS
  # backend_ext config here: its devices/cores section embeds a perf_profile
  # (burst) into the binary, and this device's unsigned PD cannot apply power
  # configs — the load then fails (createFromBinary error 1002).
  "$QAIRT_U/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe" \
      --dlc_path "$(cygpath -w "$dlc")" \
      --backend "$QAIRT_W\\lib\\x86_64-windows-msvc\\QnnHtp.dll" \
      --binary_file "${tag}_ctx" \
      --output_dir "$(cygpath -w "$ctxdir")" \
      --htp_socs sm8350 > "$ctxdir/gen.log" 2>&1
  bin=$(ls "$ctxdir"/*.bin 2>/dev/null | head -1)
  [ -n "$bin" ] || { echo "$tag: CTX FAILED (see ctx/$tag/gen.log)" >> "$SUMMARY"; continue; }

  "$QAIRT_U/bin/x86_64-windows-msvc/qnn-context-binary-utility.exe" \
      --context_binary "$(cygpath -w "$bin")" \
      --json_file "$(cygpath -w "$ctxdir/meta.json")" > /dev/null 2>&1

  out="$MODELS_OUT/zd_${tag}.onnx"
  "$SYSPY" "$BUILD/wrap_epctx.py" "$(cygpath -w "$bin")" "$(cygpath -w "$src")" "$(cygpath -w "$out")" "$tag" >> "$SUMMARY" 2>&1 \
      || { echo "$tag: WRAP FAILED" >> "$SUMMARY"; continue; }
  t1=$(date +%s)
  echo "$tag: OK ($(du -k "$bin" | cut -f1) KB ctx, $((t1-t0)) s)" | tee -a "$SUMMARY"
done

echo "ALL DONE" | tee -a "$SUMMARY"
ls -la "$MODELS_OUT" >> "$SUMMARY"
