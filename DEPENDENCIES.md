# Vendored runtime inventory

These files were copied into this standalone repository so it has no dependency
on the XREAL project tree. SHA-256 values document the exact known-working set.

| File | Purpose | SHA-256 |
|---|---|---|
| `app/src/main/assets/zipdepth.onnx` | SM8350/V68 ZipDepth EPContext model | `8C4B7BAFF7E72022D4CBFED717FD16F6357EFFFD79D41E9DCBFE4D8DB59155DF` |
| `app/libs/onnxruntime-android-qnn-1.27.0.aar` | ORT Android with QNN EP | `D814A4927C78439DA4FE599866C980EA853C2D3ECBB7078F897D559D63CCC872` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtp.so` | QNN HTP host backend, QAIRT 2.48 | `4EAA10F59FCE051E32012D6B4399C0576F5332C23349B6CC7452F9DCF8F270C7` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpPrepare.so` | QNN graph preparation support, QAIRT 2.48 | `3E408206C9F3F24F60991476EFDFF388A271FF06411C18D02660A6CEAC24CD0A` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpV68Skel.so` | Hexagon V68 DSP skeleton, QAIRT 2.48 | `C479454DC21DC1EE2995AAF922B37882684306F864259EA3C582AA2CFB2F81CA` |
| `app/src/main/jniLibs/arm64-v8a/libQnnHtpV68Stub.so` | Hexagon V68 host stub, QAIRT 2.48 | `9906A74657BE4C988A93863211D93A392E53266A15FCFD94B9809BE366AD236B` |
| `app/src/main/jniLibs/arm64-v8a/libQnnSystem.so` | QNN context metadata/runtime, QAIRT 2.48 | `7EE62754B67A1F0F3B1DEFC1C441FF59D5ED4A02BB34F9437DEF7B7C8651062D` |

The ORT C API header under `app/src/main/cpp/ort/` was extracted from the
vendored 1.27.0 AAR rather than copied from a different ORT release.
