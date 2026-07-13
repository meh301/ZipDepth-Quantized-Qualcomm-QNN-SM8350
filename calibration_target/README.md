# On-screen camera calibration target

This target has **10 × 7 inner corners** (11 × 8 squares), matching the app's
calibration detector.

1. Run **`python show_calibration_target.py`**, or double-click
   **`show-calibration-target.cmd`**. The Python script draws the target directly
   in a borderless fullscreen window. Press **Esc** or **Q** to close it. Keep
   screen brightness high and avoid reflections or visible moiré.

   Use `python show_calibration_target.py --list-monitors` to list displays and
   `python show_calibration_target.py --monitor 1` to select one. `index.html`
   remains as a browser-based fallback.
2. In the Android app, select the camera to calibrate and tap **Calibrate camera**.
   Depth inference pauses, but the RGB preview and camera capture remain live.
3. Keep the complete checkerboard visible. Move it throughout the image: center,
   left, right, high, low, close, far, and at several horizontal and vertical
   tilts. Pause briefly at each distinct view.
4. The app accepts modest camera motion while still rejecting effectively
   identical observations. It calibrates after 12 good views with adequate
   image coverage, rejects geometric outliers, and
   saves only a result with plausible intrinsics and reprojection RMS ≤ 1.5 px.
5. Repeat after choosing another camera. Results are stored separately by camera
   ID and YUV stream resolution in the app's private `files/camera-calibration/`
   directory.

The calibrated coordinates are those of the unrotated YUV camera stream. The
checker square's physical size does not affect pixel intrinsics; it only sets the
scale of the board-to-camera translation estimated during calibration.
