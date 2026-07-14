"""Build the bundled synthetic-accuracy testset for the ZipDepth NPU demo.

Takes N frames from the xreal PHOTOPAIRS captures (read-only), converts each
left half to the exact 384x384 RGB888 frame the app's camera path produces
(center-crop square + INTER_AREA resize, gray replicated to RGB), and runs the
fp32 ONNX on CPU with the app's exact preprocessing (RGB/255, NCHW) to produce
the float32 reference depth each on-device variant is scored against.

Outputs (committed as app assets):
  app/src/main/assets/testset/img_XX.rgb   384*384*3 bytes, RGB888
  app/src/main/assets/testset/ref_XX.f32   384*384 float32 LE, fp32 CPU depth

Usage: python tools/make_testset.py [fp32_onnx] [photopairs_dir] [count]
"""
import os
import sys
import glob

import cv2
import numpy as np
import onnxruntime as ort

RES = 384
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "app", "src", "main", "assets", "testset")

fp32 = sys.argv[1] if len(sys.argv) > 1 else None
pairs_dir = sys.argv[2] if len(sys.argv) > 2 else None
count = int(sys.argv[3]) if len(sys.argv) > 3 else 6

if not fp32 or not os.path.isfile(fp32):
    raise SystemExit("pass the fp32 zipdepth_base_384x384.onnx path as argv[1]")
if not pairs_dir or not os.path.isdir(pairs_dir):
    raise SystemExit("pass the PHOTOPAIRS directory as argv[2]")

os.makedirs(OUT, exist_ok=True)
session = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

frames = sorted(glob.glob(os.path.join(pairs_dir, "*.png")))
if not frames:
    raise SystemExit("no PHOTOPAIRS frames found")
# Spread selections across the capture session for scene diversity.
step = max(1, len(frames) // count)
chosen = frames[::step][:count]

for index, path in enumerate(chosen):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit("unreadable frame: %s" % path)
    # Stereo pair image: the left half is one camera view.
    left = image[:, : image.shape[1] // 2]
    # Center-crop square then resize, matching the app's upright center-crop.
    height, width = left.shape
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    square = left[y0 : y0 + side, x0 : x0 + side]
    gray = cv2.resize(square, (RES, RES), interpolation=cv2.INTER_AREA)

    rgb = np.repeat(gray[:, :, None], 3, axis=2)  # H,W,3 uint8, r=g=b
    rgb.tofile(os.path.join(OUT, "img_%02d.rgb" % index))

    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1,3,H,W
    depth = np.squeeze(session.run([output_name], {input_name: tensor})[0]).astype(np.float32)
    depth.tofile(os.path.join(OUT, "ref_%02d.f32" % index))
    print(
        "testset[%d] %s  depth[%.4f .. %.4f]"
        % (index, os.path.basename(path), float(depth.min()), float(depth.max()))
    )

print("WROTE %d image/reference pairs -> %s" % (len(chosen), OUT))
