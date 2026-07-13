package org.zipdepth.npudemo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max

@SuppressLint("SetTextI18n")
class MainActivity : Activity(), TextureView.SurfaceTextureListener {
    private lateinit var preview: TextureView
    private lateinit var depthView: ImageView
    private lateinit var metricsView: TextView
    private lateinit var statusView: TextView

    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "ZipDepthInference").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val inferenceBusy = AtomicBoolean(false)
    private val cameraOpening = AtomicBoolean(false)
    @Volatile private var modelReady = false
    @Volatile private var resumed = false
    @Volatile private var backendName = "INITIALIZING QNN…"

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraId = ""
    private var sensorOrientation = 90
    private var cameraSize = Size(1280, 720)

    private val argbBuffers = Array(2) {
        IntArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)
    }
    private val depthBitmaps = Array(2) {
        Bitmap.createBitmap(
            ZipDepthNative.OUTPUT_WIDTH,
            ZipDepthNative.OUTPUT_HEIGHT,
            Bitmap.Config.ARGB_8888,
        )
    }
    private var outputSlot = 0
    private val nativeMetrics = FloatArray(6)

    private var cameraFps = 0.0
    private var processedFps = 0.0
    private var previousCameraTimestamp = 0L
    private var previousProcessedTimestamp = 0L
    private var droppedFrames = 0L
    private var processedFrames = 0L
    private var failedFrames = 0L
    private var startupMs = 0.0
    private val metricEma = DoubleArray(4)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE

        buildUi()
        initializeModel()
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        startCameraThread()
        if (preview.isAvailable) requestOrOpenCamera()
    }

    override fun onPause() {
        resumed = false
        closeCamera()
        stopCameraThread()
        super.onPause()
    }

    override fun onDestroy() {
        inferenceExecutor.execute { ZipDepthNative.nativeShutdown() }
        inferenceExecutor.shutdown()
        super.onDestroy()
    }

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        preview = TextureView(this).apply {
            surfaceTextureListener = this@MainActivity
            isOpaque = true
        }
        root.addView(
            preview,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        val depthCard = FrameLayout(this).apply {
            background = roundedBackground(0xDD080D12.toInt(), 2f, 0x8858E6C2.toInt())
            setPadding(dp(6), dp(6), dp(6), dp(6))
            elevation = dp(8).toFloat()
        }
        val cardSize = dp(330)
        root.addView(
            depthCard,
            FrameLayout.LayoutParams(cardSize, cardSize, Gravity.END or Gravity.BOTTOM).apply {
                setMargins(dp(16), dp(16), dp(18), dp(18))
            },
        )

        depthView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF05080C.toInt())
        }
        depthCard.addView(
            depthView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        val depthLabel = TextView(this).apply {
            text = "RELATIVE DEPTH  •  NEAR = WARM"
            setTextColor(Color.WHITE)
            textSize = 11f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(5), dp(8), dp(5))
            background = roundedBackground(0xCC05080C.toInt(), 0f, Color.TRANSPARENT)
        }
        depthCard.addView(
            depthLabel,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ),
        )

        metricsView = TextView(this).apply {
            text = "ZIPDEPTH NPU DEMO\nWaiting for camera…"
            setTextColor(Color.WHITE)
            textSize = 14f
            typeface = Typeface.MONOSPACE
            setLineSpacing(0f, 1.12f)
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = roundedBackground(0xCC05080C.toInt(), 1f, 0x444A6A73)
            elevation = dp(8).toFloat()
        }
        root.addView(
            metricsView,
            FrameLayout.LayoutParams(dp(390), FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.START or Gravity.TOP).apply {
                setMargins(dp(18), dp(18), dp(18), dp(18))
            },
        )

        statusView = TextView(this).apply {
            text = backendName
            setTextColor(0xFF07120F.toInt())
            textSize = 12f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(7), dp(12), dp(7))
            background = roundedBackground(0xFF8FA6A0.toInt(), 0f, Color.TRANSPARENT)
        }
        root.addView(
            statusView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.END or Gravity.TOP,
            ).apply { setMargins(dp(18), dp(18), dp(18), dp(18)) },
        )

        setContentView(root)
    }

    private fun initializeModel() {
        inferenceExecutor.execute {
            val started = System.nanoTime()
            val model = stageModel()
            val result = if (model != null) {
                ZipDepthNative.nativeInit(model.absolutePath)
            } else {
                "ERROR: unable to stage zipdepth.onnx"
            }
            startupMs = (System.nanoTime() - started) / 1_000_000.0
            backendName = result
            modelReady = !result.startsWith("ERROR")
            runOnUiThread {
                statusView.text = if (modelReady) "HEXAGON NPU • READY" else "QNN ERROR"
                statusView.setTextColor(if (modelReady) 0xFF04100D.toInt() else Color.WHITE)
                statusView.background = roundedBackground(
                    if (modelReady) 0xFF58E6C2.toInt() else 0xFFD64A55.toInt(),
                    0f,
                    Color.TRANSPARENT,
                )
                if (!modelReady) metricsView.text = result
            }
        }
    }

    private fun stageModel(): File? = try {
        val target = File(filesDir, "zipdepth-sm8350-v68.onnx")
        if (!target.exists() || target.length() != MODEL_BYTES) {
            val temporary = File(filesDir, "zipdepth.tmp")
            assets.open("zipdepth.onnx").use { source ->
                FileOutputStream(temporary).use { output -> source.copyTo(output) }
            }
            if (target.exists()) target.delete()
            if (!temporary.renameTo(target)) {
                temporary.copyTo(target, overwrite = true)
                temporary.delete()
            }
        }
        target
    } catch (_: Exception) {
        null
    }

    private fun requestOrOpenCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION)
        } else {
            openCamera()
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        if (!resumed || cameraDevice != null || cameraHandler == null || !preview.isAvailable) return
        if (!cameraOpening.compareAndSet(false, true)) return
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            selectMainRearCamera(manager)
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpening.set(false)
                    if (!resumed) {
                        camera.close()
                        return
                    }
                    cameraDevice = camera
                    createCaptureSession(camera)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpening.set(false)
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpening.set(false)
                    camera.close()
                    cameraDevice = null
                    runOnUiThread { metricsView.text = "CAMERA ERROR $error" }
                }
            }, cameraHandler)
        } catch (error: Exception) {
            cameraOpening.set(false)
            metricsView.text = "CAMERA OPEN FAILED\n${error.message}"
        }
    }

    private fun selectMainRearCamera(manager: CameraManager) {
        val candidates = manager.cameraIdList.mapNotNull { id ->
            val characteristics = manager.getCameraCharacteristics(id)
            if (characteristics.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) {
                null
            } else {
                val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
                val logical = Build.VERSION.SDK_INT >= 28 &&
                    capabilities.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)
                val score = (if (id == "0") 1_000_000_000L else 0L) +
                    (if (logical) 100_000_000L else 0L) +
                    ((active?.width()?.toLong() ?: 0L) * (active?.height()?.toLong() ?: 0L))
                Triple(id, characteristics, score)
            }
        }
        val selected = candidates.maxByOrNull { it.third }
            ?: error("No rear RGB camera found")
        cameraId = selected.first
        val characteristics = selected.second
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        cameraSize = chooseCommonSize(
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: error("Camera has no stream configuration"),
        )
    }

    private fun chooseCommonSize(map: StreamConfigurationMap): Size {
        val textureSizes = map.getOutputSizes(SurfaceTexture::class.java)?.toSet().orEmpty()
        val yuvSizes = map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)?.toList().orEmpty()
        val common = yuvSizes.filter { textureSizes.contains(it) }.ifEmpty { yuvSizes }
        return common.minByOrNull { size ->
            val areaPenalty = abs(size.width.toLong() * size.height - 1280L * 720L)
            val aspectPenalty = abs(size.width.toDouble() / size.height - 16.0 / 9.0) * 10_000_000.0
            val oversizePenalty = if (size.width > 1920 || size.height > 1080) 1_000_000_000.0 else 0.0
            areaPenalty + aspectPenalty + oversizePenalty
        } ?: Size(1280, 720)
    }

    private fun createCaptureSession(camera: CameraDevice) {
        val texture = preview.surfaceTexture ?: return
        texture.setDefaultBufferSize(cameraSize.width, cameraSize.height)
        val previewSurface = Surface(texture)
        imageReader?.close()
        imageReader = ImageReader.newInstance(
            cameraSize.width,
            cameraSize.height,
            android.graphics.ImageFormat.YUV_420_888,
            3,
        ).also { reader ->
            reader.setOnImageAvailableListener({ onImageAvailable(it) }, cameraHandler)
        }
        val analysisSurface = imageReader!!.surface

        camera.createCaptureSession(
            listOf(previewSurface, analysisSurface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    if (cameraDevice == null) return
                    captureSession = session
                    val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(previewSurface)
                        addTarget(analysisSurface)
                        set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                        set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                        selectFpsRange()?.let { set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
                    }
                    session.setRepeatingRequest(request.build(), null, cameraHandler)
                    configurePreviewTransform()
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    runOnUiThread { metricsView.text = "CAMERA SESSION CONFIGURATION FAILED" }
                }
            },
            cameraHandler,
        )
    }

    private fun selectFpsRange(): Range<Int>? {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val ranges = manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?: return null
        return ranges.filter { it.lower <= 30 && it.upper >= 30 }
            .minByOrNull { (it.upper - it.lower) * 100 + abs(it.upper - 30) }
            ?: ranges.maxByOrNull { it.upper }
    }

    private fun onImageAvailable(reader: ImageReader) {
        val image = reader.acquireLatestImage() ?: return
        updateCameraFps(image.timestamp)
        if (!modelReady) {
            image.close()
            return
        }
        if (!inferenceBusy.compareAndSet(false, true)) {
            droppedFrames++
            image.close()
            return
        }
        try {
            inferenceExecutor.execute {
                try {
                    processImage(image)
                } finally {
                    image.close()
                    inferenceBusy.set(false)
                }
            }
        } catch (_: Exception) {
            image.close()
            inferenceBusy.set(false)
        }
    }

    private fun processImage(image: Image) {
        val planes = image.planes
        if (planes.size < 3) return
        val slot = outputSlot
        outputSlot = (outputSlot + 1) % argbBuffers.size
        val rotation = relativeImageRotation()
        val started = System.nanoTime()
        val ok = ZipDepthNative.nativeProcess(
            planes[0].buffer,
            planes[1].buffer,
            planes[2].buffer,
            image.width,
            image.height,
            planes[0].rowStride,
            planes[1].rowStride,
            planes[2].rowStride,
            planes[1].pixelStride,
            planes[2].pixelStride,
            planes[0].buffer.position(),
            planes[1].buffer.position(),
            planes[2].buffer.position(),
            rotation,
            argbBuffers[slot],
            nativeMetrics,
        )
        if (!ok) {
            failedFrames++
            runOnUiThread {
                statusView.text = "QNN RUN ERROR"
                statusView.setTextColor(Color.WHITE)
                statusView.background = roundedBackground(0xFFD64A55.toInt(), 0f, Color.TRANSPARENT)
                metricsView.text = "QNN inference failed\nSee logcat tag ZipDepthDemo for the ORT/QNN error."
            }
            return
        }

        for (index in 0..3) {
            metricEma[index] = ema(metricEma[index], nativeMetrics[index].toDouble(), 0.12)
        }

        val bitmap = depthBitmaps[slot]
        bitmap.setPixels(
            argbBuffers[slot],
            0,
            ZipDepthNative.OUTPUT_WIDTH,
            0,
            0,
            ZipDepthNative.OUTPUT_WIDTH,
            ZipDepthNative.OUTPUT_HEIGHT,
        )
        val now = System.nanoTime()
        if (previousProcessedTimestamp != 0L) {
            val instantaneous = 1_000_000_000.0 / (now - previousProcessedTimestamp).coerceAtLeast(1L)
            processedFps = ema(processedFps, instantaneous, 0.12)
        }
        previousProcessedTimestamp = now
        processedFrames++
        val pipelineMs = (now - started) / 1_000_000.0
        runOnUiThread {
            depthView.setImageBitmap(bitmap)
            renderMetrics(pipelineMs)
        }
    }

    private fun renderMetrics(pipelineMs: Double) {
        val npuMs = metricEma[2]
        val theoreticalFps = if (npuMs > 0.0) 1000.0 / npuMs else 0.0
        val warmup = max(0L, 5L - processedFrames)
        metricsView.text = String.format(
            Locale.US,
            "ZIPDEPTH • PHONE RGB CAMERA\n" +
                "Observed       %5.1f FPS\n" +
                "NPU inference  %5.1f ms  (%5.1f FPS max)\n" +
                "Pre / post     %5.1f / %5.1f ms\n" +
                "Native total   %5.1f ms   pipeline %5.1f ms\n" +
                "Camera         %5.1f FPS  %dx%d\n" +
                "Frames         %,d shown  %,d dropped  %,d failed\n" +
                "Depth window   %.4f … %.4f%s\n" +
                "Backend        %s\n" +
                "Startup        %.0f ms",
            processedFps,
            npuMs,
            theoreticalFps,
            metricEma[1],
            metricEma[3],
            metricEma[0],
            pipelineMs,
            cameraFps,
            cameraSize.width,
            cameraSize.height,
            processedFrames,
            droppedFrames,
            failedFrames,
            nativeMetrics[4],
            nativeMetrics[5],
            if (warmup > 0) "  •  warm-up $warmup" else "",
            backendName,
            startupMs,
        )
    }

    private fun updateCameraFps(timestamp: Long) {
        if (previousCameraTimestamp != 0L) {
            val instantaneous = 1_000_000_000.0 / (timestamp - previousCameraTimestamp).coerceAtLeast(1L)
            if (instantaneous in 1.0..240.0) cameraFps = ema(cameraFps, instantaneous, 0.08)
        }
        previousCameraTimestamp = timestamp
    }

    private fun relativeImageRotation(): Int {
        val displayDegrees = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation - displayDegrees + 360) % 360
    }

    private fun configurePreviewTransform() {
        if (!preview.isAvailable || preview.width == 0 || preview.height == 0) return
        val rotation = relativeImageRotation()
        val viewWidth = preview.width.toFloat()
        val viewHeight = preview.height.toFloat()
        val rotatedWidth = if (rotation == 90 || rotation == 270) cameraSize.height.toFloat() else cameraSize.width.toFloat()
        val rotatedHeight = if (rotation == 90 || rotation == 270) cameraSize.width.toFloat() else cameraSize.height.toFloat()
        val scale = max(viewWidth / rotatedWidth, viewHeight / rotatedHeight)
        val matrix = Matrix()
        val source = RectF(0f, 0f, rotatedWidth, rotatedHeight)
        val destination = RectF(
            (viewWidth - rotatedWidth * scale) * 0.5f,
            (viewHeight - rotatedHeight * scale) * 0.5f,
            (viewWidth + rotatedWidth * scale) * 0.5f,
            (viewHeight + rotatedHeight * scale) * 0.5f,
        )
        matrix.setRectToRect(source, destination, Matrix.ScaleToFit.FILL)
        if (rotation != 0) matrix.postRotate(rotation.toFloat(), viewWidth * 0.5f, viewHeight * 0.5f)
        preview.setTransform(matrix)
    }

    private fun startCameraThread() {
        if (cameraThread != null) return
        cameraThread = HandlerThread("RgbCamera").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)
    }

    private fun stopCameraThread() {
        cameraThread?.quitSafely()
        try {
            cameraThread?.join()
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        cameraThread = null
        cameraHandler = null
    }

    private fun closeCamera() {
        captureSession?.close()
        captureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else if (requestCode == CAMERA_PERMISSION) {
            metricsView.text = "CAMERA PERMISSION IS REQUIRED"
        }
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        requestOrOpenCamera()
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        configurePreviewTransform()
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        closeCamera()
        return true
    }
    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    private fun roundedBackground(color: Int, strokeDp: Float, strokeColor: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(10).toFloat()
            if (strokeDp > 0f) setStroke(dp(strokeDp.toInt().coerceAtLeast(1)), strokeColor)
        }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()
    private fun ema(previous: Double, value: Double, alpha: Double): Double =
        if (previous == 0.0) value else previous + alpha * (value - previous)

    companion object {
        private const val CAMERA_PERMISSION = 41
        private const val MODEL_BYTES = 6_939_437L
    }
}
