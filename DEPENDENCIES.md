# Vendored runtime inventory

The hardware-specific model and runtime artifacts are committed through Git LFS.
Their hashes document the exact set used for the tested Snapdragon 888 build.
These files must remain version-matched: replacing one QNN library independently
can cause initialization errors, incorrect graph placement, or a native crash.

| File | Size | Purpose | SHA-256 |
|---|---:|---|---|
| `app/src/main/assets/zipdepth.onnx` | 6.62 MiB | INT8 SM8350/v68 ZipDepth EPContext model | `8C4B7BAFF7E72022D4CBFED717FD16F6357EFFFD79D41E9DCBFE4D8DB59155DF` |
| `app/libs/onnxruntime-android-qnn-1.27.0.aar` | 7.64 MiB | ONNX Runtime 1.27.0 Android AAR with QNN EP | `D814A4927C78439DA4FE599866C980EA853C2D3ECBB7078F897D559D63CCC872` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtp.so` | 3.59 MiB | QAIRT 2.48 QNN HTP host backend | `4EAA10F59FCE051E32012D6B4399C0576F5332C23349B6CC7452F9DCF8F270C7` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpPrepare.so` | 83.84 MiB | QAIRT 2.48 HTP graph preparation support | `3E408206C9F3F24F60991476EFDFF388A271FF06411C18D02660A6CEAC24CD0A` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpV68Skel.so` | 9.77 MiB | QAIRT 2.48 Hexagon v68 DSP skeleton | `C479454DC21DC1EE2995AAF922B37882684306F864259EA3C582AA2CFB2F81CA` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpV68Stub.so` | 0.71 MiB | QAIRT 2.48 Hexagon v68 host stub | `9906A74657BE4C988A93863211D93A392E53266A15FCFD94B9809BE366AD236B` |
| `app/src/main/jniLibs/arm64-v8a/libQnnSystem.so` | 3.87 MiB | QAIRT 2.48 QNN context metadata/runtime | `7EE62754B67A1F0F3B1DEFC1C441FF59D5ED4A02BB34F9437DEF7B7C8651062D` |

The ONNX Runtime C headers under `app/src/main/cpp/ort/` match the vendored
1.27.0 AAR. OpenCV 4.12.0 is resolved from Maven Central and is therefore not
listed as a vendored binary.

## Model variants

Runtime-selectable SM8350/v68 EPContext variants produced by the pipeline in
[`tools/variants/`](tools/variants/) (BatchNorm folding + QNN-specific QDQ
quantization, then QAIRT 2.48 `qairt-converter --preserve_io_datatype` and
`qnn-context-binary-generator --htp_socs sm8350`). A16/A8 is the activation bit
width; weights are signed INT8 per-channel throughout. The A8 variants exist for
speed measurement only — see the README warning about their output quality.

| File | Size | Variant | SHA-256 |
|---|---:|---|---|
| `app/src/main/assets/models/zd_a16w8_320.onnx` | 6.55 MiB | A16W8, 320 x 320 | `07AF2271363FB2F84AD71EBBB6F0BFDBF11E41D05CA1B7B6058F63936C748D7C` |
| `app/src/main/assets/models/zd_a16w8_288.onnx` | 6.49 MiB | A16W8, 288 x 288 | `B71D5762EAD26AD72F5E78D6D837C3A203432A3F4C9DC2A316A08C90605FA42C` |
| `app/src/main/assets/models/zd_a8w8_384.onnx` | 6.37 MiB | A8W8, 384 x 384 | `F042E853701E61BA776BAF3A0FA40661988345DC3B7C3C6A57DF0074AEEF2007` |
| `app/src/main/assets/models/zd_a8w8_320.onnx` | 6.33 MiB | A8W8, 320 x 320 | `C4F9D4C70D98E519A7B21F7F23BEBD3E8D5C0415359AAC5E6C2E6B4FD71DCE3F` |
| `app/src/main/assets/models/zd_a8w8_288.onnx` | 6.31 MiB | A8W8, 288 x 288 | `4CC2DF508F2BEC45FC21B57214C96ADAE405D6E1F0583C7868062DDA2A409048` |
| `app/src/main/assets/models/zd_a8w8_256.onnx` | 6.28 MiB | A8W8, 256 x 256 | `B59E3D1A60056DD7EC4D27DD5FB4F99AAD038D8EE7554E8F1BA8CDB2C3A580C5` |

`app/src/main/assets/testset/` holds six 384 x 384 RGB888 frames (`img_XX.rgb`)
and float32 reference depth maps (`ref_XX.f32`) generated from the unquantized
fp32 ONNX on CPU by `tools/make_testset.py`. The in-app accuracy test scores the
active variant against these references.

## Licensing and redistribution

The original app code is 0BSD-licensed; ZipDepth and ONNX Runtime use the MIT
License. The Qualcomm QAIRT/QNN objects remain proprietary and are not
relicensed by this repository.

Qualcomm's AI Stack License permits object-code distribution when the QAIRT
software is incorporated into an application, but prohibits standalone QAIRT
distribution. These files are present only as components of this Android app.
Keep their notices intact, do not publish them as a separate runtime or SDK, and
review the agreement supplied with the exact SDK package before substituting
artifacts. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`LICENSES/Qualcomm-QAIRT-TERMS.md`](LICENSES/Qualcomm-QAIRT-TERMS.md).
