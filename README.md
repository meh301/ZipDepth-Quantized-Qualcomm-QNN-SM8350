# ZipDepth NPU RGB Camera Demo

Standalone Android demonstration of ZipDepth running on the Snapdragon 888
Hexagon NPU. It can use any Camera2 camera that exposes a YUV RGB stream; no XREAL hardware,
USB camera code, Basalt, SLAM, stereo processing, or parent-project files are
referenced at build or runtime.

## What the demo shows

The screen is a portrait split: the **top half is the selected live RGB camera
feed**, the **bottom half is the ZipDepth depth map** the NPU produces from it.
Both halves are rendered from the *same* upright, center-cropped 384×384 square —
the exact tensor the model consumes — so the RGB feed and the depth map are
framed identically and always correctly oriented, with no view-transform math.

- Live selectable RGB camera feed (rotated upright natively, not via a TextureView
  transform).
- Camera selector showing Camera2 ID, facing direction, focal length, and the
  full-sensor-aspect analysis resolution. Depth-only entries are disabled.
- Live 384×384 relative-depth visualization with a stable percentile window.
- Live camera FPS and delivered depth FPS.
- QNN/HTP `OrtRun`, complete native processing, dispatch round-trip, and
  capture-to-result latency.
- A visible `HTP WAIT` timer if a camera/HVX pipeline stalls Qualcomm FastRPC.

The RGB feed is painted on the camera thread for every frame, independent of the
NPU, so the camera is visibly live even while the model initializes (or if init
fails). Depth uses latest-frame backpressure: stale analysis frames are dropped
while the NPU worker is busy.

If the NPU cannot initialize, the depth half shows the exact error instead of a
silent "waiting" state; the RGB feed keeps running. Check `adb logcat -s
ZipDepthDemo onnxruntime` for the underlying ORT/QNN reason. If depth ever reads
inverted (far surfaces warm), flip the single `1.0f -` in the postprocess loop
of `zipdepth_jni.c`.

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

Grant camera permission when prompted and hold the phone in portrait: the top
half shows the live RGB feed, the bottom half shows the live ZipDepth depth map.

## Isolation from the XREAL project

The model, QNN libraries, ONNX Runtime AAR, ORT C header, Gradle wrapper, and all
application sources are copied into this repository. No Gradle project
dependency, symbolic link, relative include path, or runtime lookup points at
the XREAL repository.

`local.properties` is intentionally ignored because it contains the local
Android SDK path. Android Studio recreates it automatically on another machine.

## Interpretation of the metrics

- **CAM** is the delivered Camera2 frame rate.
- **DEPTH** is the delivered depth-map rate after latest-frame backpressure.
- **HTP** measures only `OrtApi::Run` on the QNN EPContext graph.
- **NATIVE** includes RGB tensor conversion, HTP execution, percentile
  normalization, colormapping, and JNI output copying in the NPU process.
- **ROUND** covers shared-memory dispatch, native processing, and result IPC.
- **E2E** starts before the selected camera frame is converted and copied, and
  ends when its depth result reaches the activity.

The selected analysis stream follows the active sensor's native aspect ratio.
The app requests 1.0x zoom, disables electronic and optical stabilization, then
center-crops the largest square exactly once and resizes it to the model's
384x384 input. If Camera2 exposes real-time distortion correction, the app
requests `DISTORTION_CORRECTION_MODE_FAST`; it also logs the selected camera's
reported intrinsic calibration and distortion coefficients.

Depth values are relative/non-metric. The display uses the 2nd–98th percentile
range with temporal smoothing; nearby surfaces are mapped to warm colors.

## Redistribution note

The repository vendors Qualcomm QAIRT runtime binaries and a compiled model
artifact copied from the local development environment. Review the applicable
Qualcomm and model licenses before publishing this repository or distributing
the APK outside the development team.
