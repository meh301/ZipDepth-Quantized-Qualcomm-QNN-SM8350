# Model variant build pipeline

Reproduces the bundled quantization/resolution variants in
`app/src/main/assets/models/` from the fp32 ZipDepth ONNX exports. This is the
same pipeline that produced the default `assets/zipdepth.onnx` (verified: it
regenerates that context binary byte-identically from the original DLC).

> These scripts reference machine-local paths (the QAIRT SDK install, the
> ZipDepth checkpoint clone, and helper Python environments) and are committed
> for reproducibility and documentation, not as a portable build system.

## Pipeline

1. **`zd_qnnquant.py`** — fp32 ONNX → QDQ ONNX per variant.
   - Folds every standalone `BatchNormalization` into `Mul`+`Add` first. QNN
     HTP rejects BatchNorm combined with the signed per-channel Conv QDQ this
     model needs (error 3110).
   - Quantizes with `get_qnn_qdq_config` (the QNN-EP-specific config from
     `onnxruntime.quantization.execution_providers.qnn`): MinMax calibration,
     per-channel signed INT8 weights, QUInt16 (A16) or QUInt8 (A8) activations.
   - Calibration input mimics the runtime feed: grayscale photos replicated to
     three channels, /255, with crop/flip/brightness/gamma/noise augmentation
     plus synthetic range-pinning patterns.
2. **`compile_variants.sh`** — QDQ ONNX → HTP context → app asset per variant:
   - `qairt-converter -i qdq.onnx -o x.dlc --preserve_io_datatype`
     (keeps float32 graph I/O; the interior is entirely fixed-point).
   - `qnn-context-binary-generator --dlc_path x.dlc --backend QnnHtp.dll
     --htp_socs sm8350` (QAIRT 2.48).
   - `wrap_epctx.py` embeds the binary in an ONNX `EPContext` node with the
     same schema as the default asset.
3. **`check_meta.py`** — validates every produced binary against the properties
   the known-working default has: `socModel = 0` and float32 `image`/`depth`
   graph I/O.

## Hard-won constraints (violating any of these produces binaries that fail
`QnnContext_createFromBinary` with error 1002 on device)

- **No float segments in the QDQ graph.** Default `quantize_static` leaves
  batch norms/sub/div in float; float ops compile to fp16 HTP ops, and the
  Hexagon **v68 skel has no fp16 support**.
- **No `soc_id` stamping and no perf-profile config at generation.** A
  backend-extension config with `soc_id: 30` or a `cores`/`perf_profile`
  section produces a binary the runtime refuses to load from an **unsigned
  protection domain** (which cannot query the SoC id or apply power votes).
  Use `--htp_socs sm8350` only.
- **`--preserve_io_datatype` only on a fully-quantized graph.** On a graph
  with float segments the converter emits unpreparable fp32 ops instead.

## Testset

`tools/make_testset.py` builds the accuracy-test assets
(`app/src/main/assets/testset/`): six 384x384 RGB888 frames plus fp32 CPU
reference depth maps produced by the unquantized ONNX with the app's exact
preprocessing.
