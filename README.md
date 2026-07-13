# ZipDepth NPU RGB Camera Demo

Standalone Android demonstration of ZipDepth running on the Snapdragon 888
Hexagon NPU. It uses only the phone's main rear RGB camera; no XREAL hardware,
USB camera code, Basalt, SLAM, stereo processing, or parent-project files are
referenced at build or runtime.

## What the demo shows

- Live rear RGB camera preview.
- Live 384×384 relative-depth visualization with a stable percentile window.
- Observed processed FPS and camera FPS.
- QNN/HTP inference time and its theoretical maximum FPS.
- RGB preprocessing, depth postprocessing, native total, and end-to-end
  pipeline timing.
- Presented, dropped, and failed frame counts.
- QNN initialization time and the active backend mode.

The inference loop uses latest-frame backpressure: camera preview remains fluid,
and stale analysis frames are discarded if the NPU worker is still busy.

## Target

This binary is deliberately narrow and reproducible:

- Android arm64-v8a
- Snapdragon 888 / SM8350 / Hexagon V68 (`lahaina`)
- ORT-QNN 1.27.0 as the execution-provider driver
- QAIRT/QNN 2.48 runtime libraries
- ZipDepth 384×384 INT8 QNN context wrapped in an ONNX EPContext model

The native layer rejects another SoC instead of silently falling back to CPU.
That makes all displayed inference metrics honest NPU measurements.

## Build

Open this directory as an Android Studio project, or build from PowerShell:

```powershell
.\gradlew.bat assembleDebug
```

The command-line build requires JDK 17 or newer. Android Studio's bundled JBR
works without additional configuration.

The APK is produced at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install it with:

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Grant camera permission when prompted and hold the phone in landscape.

## Isolation from the XREAL project

The model, QNN libraries, ONNX Runtime AAR, ORT C header, Gradle wrapper, and all
application sources are copied into this repository. No Gradle project
dependency, symbolic link, relative include path, or runtime lookup points at
the XREAL repository.

`local.properties` is intentionally ignored because it contains the local
Android SDK path. Android Studio recreates it automatically on another machine.

## Interpretation of the metrics

- **Observed FPS** is the real delivered depth-map rate, including camera
  acquisition and scheduling.
- **NPU inference** measures only `OrtApi::Run` on the QNN EPContext graph.
- **FPS max** is `1000 / NPU inference milliseconds`; it is a compute ceiling,
  not the camera-limited application rate.
- **Native total** includes YUV→RGB preprocessing, NPU execution, percentile
  normalization, colormapping, and the JNI pixel copy.
- **Pipeline** is measured around the entire native call and Bitmap update on
  the inference worker.
- **Dropped** counts intentionally discarded stale frames; this is expected
  whenever camera FPS exceeds inference throughput.

Depth values are relative/non-metric. The display uses the 2nd–98th percentile
range with temporal smoothing; nearby surfaces are mapped to warm colors.

## Redistribution note

The repository vendors Qualcomm QAIRT runtime binaries and a compiled model
artifact copied from the local development environment. Review the applicable
Qualcomm and model licenses before publishing this repository or distributing
the APK outside the development team.
