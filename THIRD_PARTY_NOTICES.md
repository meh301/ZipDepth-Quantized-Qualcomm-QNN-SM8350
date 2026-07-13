# Third-party notices

This project integrates or depends on the components below. Product and company
names are used only to identify interoperability targets; no affiliation or
endorsement is implied.

## ZipDepth

- Project: [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth)
- Authors: Fabio Tosi, Luca Bartolomei, Matteo Poggi, Stefano Mattoccia
- License: MIT
- Use here: source model exported, quantized, and compiled into the bundled QNN
  EPContext artifact
- License copy: [`LICENSES/ZipDepth-MIT.txt`](LICENSES/ZipDepth-MIT.txt)

## ONNX Runtime

- Project: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- Copyright: Microsoft Corporation
- License: MIT
- Use here: Android runtime, C API headers, and QNN Execution Provider
- License copy: [`LICENSES/ONNXRuntime-MIT.txt`](LICENSES/ONNXRuntime-MIT.txt)

## OpenCV

- Project: [opencv/opencv](https://github.com/opencv/opencv)
- License: Apache License 2.0
- Use here: checkerboard detection and camera calibration, resolved as the
  official `org.opencv:opencv:4.12.0` Maven dependency
- License: [OpenCV 4.12.0 LICENSE](https://github.com/opencv/opencv/blob/4.12.0/LICENSE)

## Qualcomm AI Runtime / QNN

- Product: Qualcomm AI Runtime (QAIRT), QNN API and HTP backend
- Use here: host backend, graph preparation support, Hexagon v68 stub/skeleton,
  system library, and execution of the precompiled context on SM8350
- License: Qualcomm AI Stack License supplied as `LICENSE.pdf` with the QAIRT
  SDK package
- Licensing reference:
  [`LICENSES/Qualcomm-QAIRT-TERMS.md`](LICENSES/Qualcomm-QAIRT-TERMS.md)

The AI Stack License permits distribution and sublicensing of the QAIRT
software in object-code form when incorporated into an application. Standalone
redistribution is not permitted, and Qualcomm proprietary notices must remain
intact. The QAIRT/QNN objects here are incorporated solely as runtime components
of this Android application; they are excluded from the project's 0BSD license
and must not be extracted or republished as a standalone SDK or runtime.

Qualcomm's official
[`geniex-qairt-plugin`](https://github.com/qualcomm/geniex-qairt-plugin)
repository provides a public packaging reference. Its
[`third-party notices`](https://github.com/qualcomm/geniex-qairt-plugin/blob/main/THIRD_PARTY_NOTICES.md)
identify the bundled Android QAIRT prebuilt libraries as proprietary components
governed by the QAIRT SDK EULA rather than by that repository's BSD license.
