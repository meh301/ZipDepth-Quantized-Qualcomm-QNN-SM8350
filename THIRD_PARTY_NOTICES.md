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
- Terms: proprietary Qualcomm terms associated with the SDK/runtime package

The QAIRT/QNN shared objects in this repository are not covered by the MIT or
Apache licenses above. This repository makes no grant of rights to redistribute
Qualcomm software. Review the agreement under which the artifacts were obtained
before publishing or redistributing them.
