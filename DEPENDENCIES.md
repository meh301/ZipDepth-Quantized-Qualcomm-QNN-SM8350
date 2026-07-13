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
