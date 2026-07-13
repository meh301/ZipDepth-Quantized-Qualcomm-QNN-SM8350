package org.zipdepth.npudemo

import java.nio.ByteBuffer

object ZipDepthNative {
    const val OUTPUT_WIDTH = 384
    const val OUTPUT_HEIGHT = 384

    init {
        System.loadLibrary("zipdepth_demo")
    }

    /** Returns a human-readable backend description, or an ERROR-prefixed reason. */
    external fun nativeInit(modelPath: String): String

    /**
     * Converts the camera's YUV image to upright center-cropped RGB, executes
     * ZipDepth on QNN HTP, and writes a Turbo-colored relative-depth image.
     * metrics: total, preprocess, inference, postprocess, low percentile, high percentile.
     */
    external fun nativeProcess(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yRowStride: Int,
        uRowStride: Int,
        vRowStride: Int,
        uPixelStride: Int,
        vPixelStride: Int,
        yOffset: Int,
        uOffset: Int,
        vOffset: Int,
        clockwiseRotation: Int,
        outputArgb: IntArray,
        metrics: FloatArray,
    ): Boolean

    /** Camera-process conversion to the compact RGB888 tensor published through shared memory. */
    external fun nativePrepareRgb(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yRowStride: Int,
        uRowStride: Int,
        vRowStride: Int,
        uPixelStride: Int,
        vPixelStride: Int,
        yOffset: Int,
        uOffset: Int,
        vOffset: Int,
        clockwiseRotation: Int,
        outputRgb: ByteArray,
    ): Boolean

    /** Runs an already cropped RGB888 frame in the camera-free NPU service process. */
    external fun nativeProcessRgb(
        rgb: ByteArray,
        outputArgb: IntArray,
        metrics: FloatArray,
    ): Boolean

    /**
     * Cheap YUV -> upright center-cropped RGB with no NPU. Fills [outputArgb]
     * (OUTPUT_WIDTH * OUTPUT_HEIGHT) so the live preview is correctly oriented
     * and framed identically to the depth map, independent of model state.
     */
    external fun nativePreview(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yRowStride: Int,
        uRowStride: Int,
        vRowStride: Int,
        uPixelStride: Int,
        vPixelStride: Int,
        yOffset: Int,
        uOffset: Int,
        vOffset: Int,
        clockwiseRotation: Int,
        outputArgb: IntArray,
    ): Boolean

    external fun nativeShutdown()
}
