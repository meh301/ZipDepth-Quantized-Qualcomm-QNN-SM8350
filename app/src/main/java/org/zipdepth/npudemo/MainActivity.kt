package org.zipdepth.npudemo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

// ZipDepth NPU demo: selected RGB camera on top, live NPU depth on the bottom,
// with a compact camera picker and honest live pipeline metrics.
@SuppressLint("SetTextI18n")
class MainActivity : Activity() {
    private lateinit var previewView: ImageView
    private lateinit var depthView: ImageView
    private lateinit var statusView: TextView
    private lateinit var cameraSelectorView: TextView
    private lateinit var modelSelectorView: TextView
    private lateinit var statsView: TextView
    private lateinit var accuracyButtonView: TextView
    private lateinit var calibrationButtonView: TextView
    private lateinit var calibrationStatusView: TextView
    private lateinit var calibrationController: CameraCalibrationController

    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "ZipDepthInference").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val npuBusy = AtomicBoolean(false)
    private val frameLock = Any()
    private val freeFrames = ArrayDeque<RgbFrame>(2)
    @Volatile private var latestFrame: RgbFrame? = null
    @Volatile private var npuService: Messenger? = null
    private var npuBound = false
    private var sharedFrameMap: MappedByteBuffer? = null
    private var sharedFrameFile: File? = null
    private var frameSequence = 0L
    private val cameraOpening = AtomicBoolean(false)
    @Volatile private var modelReady = false
    @Volatile private var resumed = false
    @Volatile private var depthShown = false
    @Volatile private var calibrationActive = false

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraId = ""
    private var selectedCameraId: String? = null
    private var cameraOptions = emptyList<CameraOption>()
    @Volatile private var cameraGeneration = 0
    private var sensorOrientation = 90
    @Volatile private var cameraSize = Size(1280, 720)

    private val previewArgb = Array(2) { IntArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT) }
    private val previewBitmaps = Array(2) {
        Bitmap.createBitmap(ZipDepthNative.OUTPUT_WIDTH, ZipDepthNative.OUTPUT_HEIGHT, Bitmap.Config.ARGB_8888)
    }
    private var previewSlot = 0

    // Depth bitmaps are cached per network resolution (model variants run at
    // 256/288/320/384); each entry is a double-buffer pair.
    private val depthBitmapCache = HashMap<Int, Array<Bitmap>>()
    private var depthSlot = 0
    private val nativeMetrics = FloatArray(6)
    private val remoteDepthArgb = IntArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)
    private val remoteDepthRaw = FloatArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)

    // ---- Model variants ----
    private var modelOptions = emptyList<ModelOption>()
    @Volatile private var selectedModel: ModelOption? = null
    @Volatile private var modelRes = ZipDepthNative.OUTPUT_WIDTH
    @Volatile private var switchRetryUsed = false

    // ---- Synthetic accuracy test ----
    @Volatile private var accuracyActive = false
    private var accuracyIndex = 0
    private var accuracyCount = 0
    private val accuracyD1 = ArrayList<Double>()
    private val accuracyRmse = ArrayList<Double>()
    private val accuracyCorr = ArrayList<Double>()
    private var accuracyRef: FloatArray? = null

    private var cameraRateStartNs = 0L
    private var cameraRateFrames = 0
    @Volatile private var cameraFps = 0f
    private var depthRateStartNs = 0L
    private var depthRateFrames = 0
    private var depthFps = 0f
    @Volatile private var inFlightCaptureNs = 0L
    @Volatile private var inFlightDispatchNs = 0L
    @Volatile private var inFlightCameraGeneration = 0
    private var pipelineLatencyMs = 0f
    private var dispatchLatencyMs = 0f
    private var statsLastDrawNs = 0L

    private val npuClient = Messenger(Handler(Looper.getMainLooper()) { message ->
        handleNpuMessage(message)
        true
    })

    private val npuConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            npuService = Messenger(binder)
            if (stagedModelPath == null) {
                setStatus("NPU MODEL PATH MISSING", true)
                return
            }
            sendModelInitialize()
        }

        override fun onServiceDisconnected(name: ComponentName) {
            // The NPU process died (crash, or a deliberate MSG_RESTART during a
            // failed model switch). AMS restarts a bound BIND_AUTO_CREATE service
            // automatically; onServiceConnected then re-sends INITIALIZE.
            npuService = null
            modelReady = false
            npuBusy.set(false)
            if (accuracyActive) finishAccuracyTest("NPU PROCESS RESTARTED")
        }
    }

    @Volatile private var stagedModelPath: String? = null

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
        calibrationController = CameraCalibrationController(this) { state ->
            runOnUiThread { applyCalibrationState(state) }
        }
        loadCameraOptions()
        loadModelOptions()
        refreshStoredCalibration()
        initializeModel()
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        startCameraThread()
        // Warm the remote HTP session before starting camera capture. Once the
        // camera starts it remains continuous; this only orders initial startup.
        if (modelReady) requestOrOpenCamera()
    }

    // Debug/automation hook: `adb shell am start --activity-single-top -n <cmp>
    // -e model_asset models/zd_a8w8_384.onnx` drives the same switchModel path
    // as the MODEL dropdown, so model switching is testable without the UI.
    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        val requested = intent?.getStringExtra(EXTRA_MODEL_ASSET) ?: return
        val option = modelOptions.firstOrNull { it.assetPath == requested }
        if (option != null) {
            Log.i(TAG, "intent-driven model switch -> ${option.assetPath}")
            switchModel(option)
        } else {
            Log.w(TAG, "intent-driven model switch: unknown asset '$requested'")
        }
    }

    override fun onPause() {
        resumed = false
        if (::calibrationController.isInitialized && calibrationController.isActive()) {
            calibrationController.cancel("CALIBRATION CANCELLED — APP LEFT FOREGROUND")
        }
        closeCamera()
        stopCameraThread()
        super.onPause()
    }

    override fun onDestroy() {
        modelReady = false
        if (npuBound) {
            unbindService(npuConnection)
            npuBound = false
        }
        npuService = null
        inferenceExecutor.shutdown()
        if (::calibrationController.isInitialized) calibrationController.close()
        super.onDestroy()
    }

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(
            column,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        previewView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        column.addView(previewView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        column.addView(
            View(this).apply { setBackgroundColor(0xFF58E6C2.toInt()) },
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(2)),
        )

        depthView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
        }
        column.addView(depthView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        statusView = TextView(this).apply {
            text = "INITIALISING NPU…"
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(dp(14), dp(8), dp(14), dp(8))
            setBackgroundColor(0xCC000000.toInt())
        }
        root.addView(
            statusView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER_HORIZONTAL or Gravity.TOP,
            ).apply { setMargins(dp(12), dp(124), dp(12), dp(12)) },
        )

        cameraSelectorView = TextView(this).apply {
            text = "CAMERA"
            setTextColor(Color.WHITE)
            textSize = 12f
            gravity = Gravity.CENTER_VERTICAL
            maxLines = 2
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(0xCC111111.toInt())
            setOnClickListener { showCameraMenu() }
        }
        root.addView(
            cameraSelectorView,
            FrameLayout.LayoutParams(dp(196), FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.START or Gravity.TOP)
                .apply { setMargins(dp(12), dp(12), dp(12), dp(12)) },
        )

        modelSelectorView = TextView(this).apply {
            text = "MODEL"
            setTextColor(Color.WHITE)
            textSize = 12f
            gravity = Gravity.CENTER_VERTICAL
            maxLines = 2
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(0xCC10261F.toInt())
            setOnClickListener { showModelMenu() }
        }
        root.addView(
            modelSelectorView,
            FrameLayout.LayoutParams(dp(196), FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.START or Gravity.TOP)
                .apply { setMargins(dp(12), dp(68), dp(12), dp(12)) },
        )

        statsView = TextView(this).apply {
            text = "CAM  --.- fps\nDEPTH --.- fps\nHTP  --.- ms\nPIPE --.- ms"
            setTextColor(Color.WHITE)
            textSize = 11f
            gravity = Gravity.END
            typeface = Typeface.MONOSPACE
            setPadding(dp(10), dp(7), dp(10), dp(7))
            setBackgroundColor(0xCC111111.toInt())
        }
        root.addView(
            statsView,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.END or Gravity.TOP)
                .apply { setMargins(dp(12), dp(12), dp(12), dp(12)) },
        )

        accuracyButtonView = TextView(this).apply {
            text = "TEST ACC"
            setTextColor(Color.WHITE)
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(dp(14), dp(8), dp(14), dp(8))
            setBackgroundColor(0xDD34406B.toInt())
            setOnClickListener { startAccuracyTest() }
        }
        root.addView(
            accuracyButtonView,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.END or Gravity.TOP)
                .apply { setMargins(dp(12), dp(132), dp(12), dp(12)) },
        )

        calibrationButtonView = TextView(this).apply {
            text = "CALIBRATE CAMERA"
            setTextColor(Color.WHITE)
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(11), dp(16), dp(11))
            setBackgroundColor(0xDD176B5B.toInt())
            setOnClickListener { toggleCalibration() }
        }
        root.addView(
            calibrationButtonView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM,
            ).apply { setMargins(dp(12), dp(12), dp(12), dp(18)) },
        )

        calibrationStatusView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 12f
            gravity = Gravity.CENTER
            typeface = Typeface.MONOSPACE
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(0xDD111111.toInt())
            visibility = View.GONE
        }
        root.addView(
            calibrationStatusView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM,
            ).apply { setMargins(dp(12), dp(12), dp(12), dp(68)) },
        )

        setContentView(root)
    }

    private fun setStatus(text: String, show: Boolean) = runOnUiThread {
        statusView.text = text
        statusView.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun toggleCalibration() {
        if (calibrationController.isActive()) {
            calibrationController.cancel()
            return
        }
        val option = cameraOptions.firstOrNull {
            it.id == selectedCameraId && it.analysisSize != null
        }
        if (option == null) {
            calibrationStatusView.text = "NO CALIBRATABLE YUV CAMERA"
            calibrationStatusView.visibility = View.VISIBLE
            return
        }
        synchronized(frameLock) {
            latestFrame?.let { freeFrames.addLast(it) }
            latestFrame = null
        }
        calibrationController.start(option.id, option.analysisSize!!.width, option.analysisSize.height)
    }

    private fun applyCalibrationState(state: CalibrationUiState) {
        val wasActive = calibrationActive
        calibrationActive = state.active
        calibrationButtonView.text = when {
            state.active -> "CANCEL CALIBRATION  ${state.acceptedViews}/${state.targetViews}"
            state.calibration != null -> "CALIBRATED · RE-CALIBRATE"
            state.failed -> "RETRY CALIBRATION"
            else -> "CALIBRATE CAMERA"
        }
        calibrationButtonView.setBackgroundColor(
            when {
                state.active -> 0xDDBB6A18.toInt()
                state.failed -> 0xDDA22B35.toInt()
                else -> 0xDD176B5B.toInt()
            },
        )
        calibrationStatusView.text = state.message
        calibrationStatusView.setBackgroundColor(
            if (state.failed) 0xE0991F2A.toInt() else 0xDD111111.toInt(),
        )
        calibrationStatusView.visibility = View.VISIBLE
        refreshStats(force = true)
        if (wasActive && !state.active && modelReady && resumed) scheduleInferenceDispatch()
    }

    private fun refreshStoredCalibration() {
        if (!::calibrationController.isInitialized || !::calibrationButtonView.isInitialized || calibrationActive) return
        val option = cameraOptions.firstOrNull {
            it.id == selectedCameraId && it.analysisSize != null
        }
        val size = option?.analysisSize
        val calibration = if (option != null && size != null) {
            calibrationController.load(option.id, size.width, size.height)
        } else {
            null
        }
        if (calibration == null) {
            calibrationButtonView.text = "CALIBRATE CAMERA"
            calibrationStatusView.visibility = View.GONE
        } else {
            calibrationButtonView.text = "CALIBRATED · RE-CALIBRATE"
            calibrationStatusView.text = calibration.summary()
            calibrationStatusView.setBackgroundColor(0xDD111111.toInt())
            calibrationStatusView.visibility = View.VISIBLE
        }
    }

    private fun loadCameraOptions() {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        cameraOptions = try {
            manager.cameraIdList.map { id ->
                val characteristics = manager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val yuvSizes = map?.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)
                    ?.toList().orEmpty()
                val activeArray = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                val sensorAspect = activeArray?.let { active ->
                    maxOf(active.width(), active.height()).toDouble() /
                        minOf(active.width(), active.height()).coerceAtLeast(1)
                }
                val analysisSize = if (map != null && yuvSizes.isNotEmpty()) {
                    chooseAnalysisSize(map, sensorAspect)
                } else {
                    null
                }
                val focalLength = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.firstOrNull()
                CameraOption(id, facing, focalLength, analysisSize)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "unable to enumerate cameras", t)
            emptyList()
        }

        val usable = cameraOptions.filter { it.analysisSize != null }
        val requested = intent.getStringExtra(EXTRA_CAMERA_ID)
        val saved = getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(PREF_CAMERA_ID, null)
        selectedCameraId = listOfNotNull(requested, saved, "0")
            .firstOrNull { id -> usable.any { it.id == id } }
            ?: usable.firstOrNull { it.facing == CameraCharacteristics.LENS_FACING_BACK }?.id
            ?: usable.firstOrNull()?.id
        updateCameraSelectorLabel()
    }

    private fun facingLabel(facing: Int?): String = when (facing) {
        CameraCharacteristics.LENS_FACING_FRONT -> "FRONT"
        CameraCharacteristics.LENS_FACING_BACK -> "REAR"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "EXTERNAL"
        else -> "UNKNOWN"
    }

    private fun cameraMenuLabel(option: CameraOption): String {
        val lens = option.focalLength?.let { String.format(Locale.US, "%.1f mm", it) } ?: "lens ?"
        val size = option.analysisSize?.let { "${it.width}x${it.height}" } ?: "NO YUV"
        return "ID ${option.id}  ${facingLabel(option.facing)}  $lens  $size"
    }

    private fun updateCameraSelectorLabel() {
        val option = cameraOptions.firstOrNull { it.id == selectedCameraId }
        cameraSelectorView.text = if (option == null) {
            "NO RGB CAMERA"
        } else {
            val size = option.analysisSize
            "CAM ${option.id}  ${facingLabel(option.facing)}\n${size?.width ?: 0}x${size?.height ?: 0}  [CHANGE]"
        }
    }

    private fun showCameraMenu() {
        val popup = PopupMenu(this, cameraSelectorView)
        cameraOptions.forEachIndexed { index, option ->
            popup.menu.add(0, index, index, cameraMenuLabel(option)).apply {
                isEnabled = option.analysisSize != null
                isCheckable = true
                isChecked = option.id == selectedCameraId
            }
        }
        popup.setOnMenuItemClickListener { item ->
            val option = cameraOptions.getOrNull(item.itemId) ?: return@setOnMenuItemClickListener false
            switchCamera(option.id)
            true
        }
        popup.show()
    }

    private fun switchCamera(id: String) {
        val option = cameraOptions.firstOrNull { it.id == id && it.analysisSize != null } ?: return
        if (selectedCameraId == id && cameraDevice != null) return
        if (calibrationController.isActive()) {
            calibrationController.cancel("CALIBRATION CANCELLED — CAMERA CHANGED")
        }
        selectedCameraId = id
        getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit()
            .putString(PREF_CAMERA_ID, id)
            .apply()
        updateCameraSelectorLabel()
        refreshStoredCalibration()
        resetLiveStats()
        synchronized(frameLock) {
            latestFrame?.let { freeFrames.addLast(it) }
            latestFrame = null
        }
        depthShown = false
        depthView.setImageDrawable(null)
        setStatus("SWITCHING TO CAMERA $id", true)
        closeCamera()
        if (resumed && modelReady) cameraHandler?.post { openCamera() }
    }

    // ---- Model variant selection ----

    /**
     * The candidate models: the shipped baseline asset plus every compiled
     * variant bundled under assets/models (zd_<quant>_<res>.onnx).
     */
    private fun loadModelOptions() {
        val options = ArrayList<ModelOption>()
        options.add(ModelOption("zipdepth.onnx", "BASE A16 384"))
        val variants = try {
            assets.list(MODELS_DIR).orEmpty().filter { it.endsWith(".onnx") }.sorted()
        } catch (t: Throwable) {
            Log.e(TAG, "unable to list bundled model variants", t)
            emptyList()
        }
        for (file in variants) {
            options.add(ModelOption("$MODELS_DIR/$file", variantLabel(file)))
        }
        modelOptions = options
        val saved = getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(PREF_MODEL_ASSET, null)
        selectedModel = options.firstOrNull { it.assetPath == saved } ?: options.first()
        updateModelSelectorLabel()
    }

    // zd_a8w8_288.onnx -> "A8W8 288 ⚠". The 8-bit-activation variants produce
    // visibly degraded, temporally flickering depth; they are bundled to measure
    // the full-INT8 speed envelope, and the marker warns against judging quality
    // by them.
    private fun variantLabel(file: String): String {
        val stem = file.removeSuffix(".onnx").removePrefix("zd_")
        val label = stem.split('_').joinToString(" ") { it.uppercase(Locale.US) }
        return if (stem.startsWith("a8")) "$label ⚠" else label
    }

    private fun storedAccuracy(option: ModelOption): String? =
        getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(PREF_ACCURACY_PREFIX + option.assetPath, null)

    private fun updateModelSelectorLabel() {
        val option = selectedModel
        modelSelectorView.text = if (option == null) {
            "NO MODEL"
        } else {
            val acc = storedAccuracy(option)?.let { "\n$it" } ?: ""
            "MODEL ${option.label}  [CHANGE]$acc"
        }
    }

    private fun showModelMenu() {
        if (accuracyActive) return
        val popup = PopupMenu(this, modelSelectorView)
        modelOptions.forEachIndexed { index, option ->
            val acc = storedAccuracy(option)?.let { "   $it" } ?: ""
            popup.menu.add(0, index, index, "${option.label}$acc").apply {
                isCheckable = true
                isChecked = option.assetPath == selectedModel?.assetPath
            }
        }
        popup.setOnMenuItemClickListener { item ->
            val option = modelOptions.getOrNull(item.itemId) ?: return@setOnMenuItemClickListener false
            switchModel(option)
            true
        }
        popup.show()
    }

    private fun switchModel(option: ModelOption) {
        if (option.assetPath == selectedModel?.assetPath && modelReady) return
        selectedModel = option
        switchRetryUsed = false
        getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit()
            .putString(PREF_MODEL_ASSET, option.assetPath)
            .apply()
        updateModelSelectorLabel()
        modelReady = false
        depthShown = false
        depthView.setImageDrawable(null)
        nativeMetrics.fill(0f)
        resetLiveStats()
        setStatus("LOADING ${option.label}…", true)
        inferenceExecutor.execute {
            val staged = stageModelAsset(option)
            if (staged == null) {
                setStatus("UNABLE TO STAGE ${option.label}", true)
                return@execute
            }
            stagedModelPath = staged.absolutePath
            sendModelInitialize()
        }
    }

    private fun sendModelInitialize() {
        val service = npuService
        val path = stagedModelPath
        if (service == null || path == null) return
        try {
            service.send(Message.obtain(null, ZipDepthService.MSG_INITIALIZE).apply {
                replyTo = npuClient
                data = Bundle().apply {
                    putString(ZipDepthService.KEY_MODEL_PATH, path)
                    putString(ZipDepthService.KEY_FRAME_PATH, sharedFrameFile?.absolutePath)
                }
            })
        } catch (t: Throwable) {
            setStatus("NPU SERVICE INIT FAILED\n${t.message}", true)
        }
    }

    private fun initializeModel() {
        inferenceExecutor.execute {
            val option = selectedModel
            val model = option?.let { stageModelAsset(it) }
            if (model == null) {
                setStatus("NPU init failed:\nunable to stage ${option?.assetPath ?: "model"} from assets", true)
                if (resumed) runOnUiThread { requestOrOpenCamera() }
                return@execute
            }
            val frameFile = prepareSharedFrameBuffer()
            if (frameFile == null) {
                setStatus("NPU init failed:\nunable to create shared frame buffer", true)
                if (resumed) runOnUiThread { requestOrOpenCamera() }
                return@execute
            }
            stagedModelPath = model.absolutePath
            runOnUiThread {
                npuBound = bindService(
                    Intent(this, ZipDepthService::class.java),
                    npuConnection,
                    Context.BIND_AUTO_CREATE,
                )
                if (!npuBound) setStatus("NPU SERVICE BIND FAILED", true)
            }
        }
    }

    private fun handleNpuMessage(message: Message) {
        when (message.what) {
            ZipDepthService.MSG_INITIALIZED -> {
                val result = message.data.getString(ZipDepthService.KEY_RESULT).orEmpty()
                val reportedRes = message.data.getInt(ZipDepthService.KEY_MODEL_RES, 0)
                modelReady = !result.startsWith("ERROR")
                if (modelReady && reportedRes > 0) modelRes = reportedRes
                Log.i(TAG, "remote model init: ready=$modelReady res=$reportedRes result=\"$result\"")
                if (modelReady) {
                    switchRetryUsed = false
                    setStatus("NPU READY • ${selectedModel?.label ?: "?"} @${modelRes}px", true)
                    updateModelSelectorLabel()
                    if (resumed) requestOrOpenCamera()
                    scheduleInferenceDispatch()
                } else if (!switchRetryUsed && npuService != null) {
                    // A failed in-process re-init can leave poisoned QNN state:
                    // restart the isolated NPU process once and retry the same
                    // model on the pristine process that AMS brings back.
                    switchRetryUsed = true
                    setStatus("MODEL LOAD FAILED — RESTARTING NPU PROCESS", true)
                    try {
                        npuService?.send(Message.obtain(null, ZipDepthService.MSG_RESTART))
                    } catch (_: Throwable) {
                    }
                } else {
                    setStatus("NPU init failed:\n$result", true)
                    if (resumed) requestOrOpenCamera()
                }
            }

            ZipDepthService.MSG_RESULT -> {
                val ok = message.data.getBoolean(ZipDepthService.KEY_OK)
                val resultRes = message.data.getInt(ZipDepthService.KEY_MODEL_RES, modelRes)
                val currentResult = inFlightCameraGeneration == cameraGeneration
                if (ok && currentResult && !calibrationActive) {
                    message.data.getFloatArray(ZipDepthService.KEY_METRICS)?.let { values ->
                        values.copyInto(nativeMetrics, endIndex = minOf(values.size, nativeMetrics.size))
                    }
                    val now = SystemClock.elapsedRealtimeNanos()
                    if (inFlightCaptureNs > 0L) {
                        pipelineLatencyMs = ((now - inFlightCaptureNs) / 1_000_000.0).toFloat()
                    }
                    if (inFlightDispatchNs > 0L) {
                        dispatchLatencyMs = ((now - inFlightDispatchNs) / 1_000_000.0).toFloat()
                    }
                    markDepthFrame(now)
                    readAndRenderSharedDepth(resultRes)
                } else if (!ok) {
                    modelReady = false
                    setStatus(message.data.getString(ZipDepthService.KEY_ERROR) ?: "QNN RUN ERROR", true)
                }
                npuBusy.set(false)
                inFlightCaptureNs = 0L
                inFlightDispatchNs = 0L
                refreshStats(force = true)
                if (modelReady) scheduleInferenceDispatch()
            }

            ZipDepthService.MSG_RESULT_RAW -> {
                val ok = message.data.getBoolean(ZipDepthService.KEY_OK)
                val resultRes = message.data.getInt(ZipDepthService.KEY_MODEL_RES, modelRes)
                message.data.getFloatArray(ZipDepthService.KEY_METRICS)?.let { values ->
                    values.copyInto(nativeMetrics, endIndex = minOf(values.size, nativeMetrics.size))
                }
                npuBusy.set(false)
                if (accuracyActive) {
                    if (ok) {
                        onAccuracyResult(resultRes)
                    } else {
                        finishAccuracyTest("TEST FAILED: " +
                            (message.data.getString(ZipDepthService.KEY_ERROR) ?: "inference error"))
                    }
                }
            }
        }
    }

    /**
     * Stages one bundled model asset into filesDir/models/. A staged copy is
     * trusted when its on-disk length matches the length recorded at the last
     * successful copy of that asset (assets may be aapt-compressed, so the
     * stream is the only reliable source of the true size).
     */
    private fun stageModelAsset(option: ModelOption): File? = try {
        val directory = File(filesDir, "models").apply { mkdirs() }
        val target = File(directory, option.assetPath.substringAfterLast('/'))
        val preferences = getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val recordedLength = preferences.getLong(PREF_STAGED_PREFIX + option.assetPath, -1L)
        if (!target.exists() || target.length() != recordedLength || recordedLength <= 0L) {
            val temporary = File(directory, "${target.name}.tmp")
            assets.open(option.assetPath).use { source ->
                FileOutputStream(temporary).use { output -> source.copyTo(output) }
            }
            if (target.exists()) target.delete()
            if (!temporary.renameTo(target)) {
                temporary.copyTo(target, overwrite = true)
                temporary.delete()
            }
            preferences.edit()
                .putLong(PREF_STAGED_PREFIX + option.assetPath, target.length())
                .apply()
        }
        target
    } catch (t: Exception) {
        Log.e(TAG, "unable to stage ${option.assetPath}", t)
        null
    }

    private fun prepareSharedFrameBuffer(): File? = try {
        val file = File(noBackupFilesDir, "zipdepth-rgb.frame")
        RandomAccessFile(file, "rw").use { randomFile ->
            randomFile.setLength(SHARED_BUFFER_BYTES.toLong())
            sharedFrameMap = randomFile.channel.map(
                FileChannel.MapMode.READ_WRITE,
                0L,
                SHARED_BUFFER_BYTES.toLong(),
            )
        }
        sharedFrameFile = file
        file
    } catch (t: Throwable) {
        Log.e(TAG, "unable to map shared RGB input", t)
        sharedFrameMap = null
        sharedFrameFile = null
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
        if (!resumed || cameraDevice != null || cameraHandler == null) return
        if (!cameraOpening.compareAndSet(false, true)) return
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            configureSelectedCamera(manager)
            val openingId = cameraId
            val generation = cameraGeneration
            manager.openCamera(openingId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpening.set(false)
                    if (!resumed || generation != cameraGeneration || openingId != selectedCameraId) {
                        camera.close()
                        return
                    }
                    cameraDevice = camera
                    createCaptureSession(camera, generation)
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
                    setStatus("CAMERA ERROR $error", true)
                }
            }, cameraHandler)
        } catch (error: Exception) {
            cameraOpening.set(false)
            setStatus("CAMERA OPEN FAILED\n${error.message}", true)
        }
    }

    private fun configureSelectedCamera(manager: CameraManager) {
        val selected = cameraOptions.firstOrNull {
            it.id == selectedCameraId && it.analysisSize != null
        } ?: error("No selectable RGB/YUV camera found")
        cameraId = selected.id
        val characteristics = manager.getCameraCharacteristics(cameraId)
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        cameraSize = selected.analysisSize ?: error("Camera $cameraId has no YUV output")
        Log.i(TAG, "selected camera id=$cameraId facing=${facingLabel(selected.facing)} size=$cameraSize")
    }

    private fun chooseAnalysisSize(map: StreamConfigurationMap, sensorAspect: Double?): Size {
        val yuvSizes = map.getOutputSizes(android.graphics.ImageFormat.YUV_420_888)?.toList().orEmpty()
        val fullSensorAspect = sensorAspect?.takeIf { it.isFinite() && it >= 1.0 }
            ?: yuvSizes.maxByOrNull { it.width.toLong() * it.height }
                ?.let { maxOf(it.width, it.height).toDouble() / minOf(it.width, it.height).coerceAtLeast(1) }
            ?: (4.0 / 3.0)
        val targetLongEdge = 1280.0
        val targetArea = targetLongEdge * (targetLongEdge / fullSensorAspect)
        // Prefer any stream which preserves the active-array aspect, even if
        // the HAL only exposes that aspect at a larger resolution. Sampling
        // directly into 384x384 is cheaper than accepting an earlier HAL crop.
        val fullAspectSizes = yuvSizes.filter { size ->
            val longEdge = maxOf(size.width, size.height)
            val shortEdge = minOf(size.width, size.height).coerceAtLeast(1)
            val aspect = longEdge.toDouble() / shortEdge
            abs(aspect - fullSensorAspect) <= 0.015
        }
        val candidates = fullAspectSizes.ifEmpty { yuvSizes }
        return candidates.minByOrNull { size ->
            val longEdge = maxOf(size.width, size.height)
            val shortEdge = minOf(size.width, size.height).coerceAtLeast(1)
            val aspect = longEdge.toDouble() / shortEdge
            val aspectPenalty = abs(aspect - fullSensorAspect) * 1_000_000_000.0
            abs(size.width.toDouble() * size.height - targetArea) + aspectPenalty
        } ?: Size(1280, 960)
    }

    private fun createCaptureSession(camera: CameraDevice, generation: Int) {
        imageReader?.close()
        val reader = ImageReader.newInstance(
            cameraSize.width,
            cameraSize.height,
            android.graphics.ImageFormat.YUV_420_888,
            3,
        ).also { it.setOnImageAvailableListener({ r -> onImageAvailable(r) }, cameraHandler) }
        imageReader = reader
        val analysisSurface = reader.surface

        camera.createCaptureSession(
            listOf(analysisSurface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    if (cameraDevice !== camera || generation != cameraGeneration) {
                        session.close()
                        return
                    }
                    captureSession = session
                    val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                        addTarget(analysisSurface)
                        set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                        set(CaptureRequest.CONTROL_ENABLE_ZSL, false)
                        selectFpsRange()?.let { set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
                        configureGeometryControls(this)
                        configureFocusAndDepthControls(this)
                    }
                    session.setRepeatingRequest(request.build(), null, cameraHandler)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    setStatus("CAMERA SESSION CONFIG FAILED", true)
                }
            },
            cameraHandler,
        )
    }

    private fun configureGeometryControls(request: CaptureRequest.Builder) {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val requestKeys = characteristics.availableCaptureRequestKeys

        var zoomControl = "full active array"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R &&
            requestKeys.contains(CaptureRequest.CONTROL_ZOOM_RATIO) &&
            characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.contains(1f) == true
        ) {
            request.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1f)
            zoomControl = "zoomRatio=1.0"
        } else {
            characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.let { activeArray ->
                if (requestKeys.contains(CaptureRequest.SCALER_CROP_REGION)) {
                    request.set(CaptureRequest.SCALER_CROP_REGION, activeArray)
                }
            }
        }

        val videoStabilizationModes = characteristics.get(
            CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES,
        ) ?: intArrayOf()
        if (requestKeys.contains(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE) &&
            videoStabilizationModes.contains(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        ) {
            request.set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
        }
        val opticalStabilizationModes = characteristics.get(
            CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION,
        ) ?: intArrayOf()
        if (requestKeys.contains(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE) &&
            opticalStabilizationModes.contains(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
        ) {
            request.set(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        }

        var distortionControl = "unavailable"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val modes = characteristics.get(
                CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES,
            ) ?: intArrayOf()
            if (modes.contains(CaptureRequest.DISTORTION_CORRECTION_MODE_FAST) &&
                requestKeys.contains(CaptureRequest.DISTORTION_CORRECTION_MODE)
            ) {
                request.set(
                    CaptureRequest.DISTORTION_CORRECTION_MODE,
                    CaptureRequest.DISTORTION_CORRECTION_MODE_FAST,
                )
                distortionControl = "FAST"
            }
            val intrinsics = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION)
            Log.i(
                TAG,
                "camera $cameraId calibration: intrinsics=${intrinsics?.contentToString() ?: "none"} " +
                    "distortion=${distortion?.contentToString() ?: "none"}",
            )
        }
        Log.i(
            TAG,
            "camera $cameraId geometry: stream=${cameraSize.width}x${cameraSize.height} " +
                "$zoomControl distortion=$distortionControl stabilization=off",
        )
    }

    private fun configureFocusAndDepthControls(request: CaptureRequest.Builder) {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val afModes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
        val sharpRear = cameraId == "0" && Build.MANUFACTURER.equals("SHARP", ignoreCase = true)
        val afMode = if (!sharpRear && afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)) {
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
        } else {
            CaptureRequest.CONTROL_AF_MODE_OFF
        }
        request.set(CaptureRequest.CONTROL_AF_MODE, afMode)
        if (afMode == CaptureRequest.CONTROL_AF_MODE_OFF &&
            characteristics.availableCaptureRequestKeys.contains(CaptureRequest.LENS_FOCUS_DISTANCE)
        ) {
            request.set(CaptureRequest.LENS_FOCUS_DISTANCE, 0f)
        }
        if (sharpRear && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                request.set(
                    CaptureRequest.Key(
                        "com.sharp.camera.tofsensor.disable",
                        Byte::class.javaObjectType,
                    ),
                    1.toByte(),
                )
                Log.i(TAG, "requested SHARP ToF camera disable")
            } catch (t: Throwable) {
                Log.w(TAG, "SHARP ToF disable vendor tag was rejected", t)
            }
        }
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
        try {
            markCameraFrame()
            // Camera analysis is the producer and never waits for QNN. The NPU sees
            // an app-owned copy through a latest-frame mailbox below.
            renderPreview(image)
            if (calibrationActive) {
                calibrationController.offer(image)
            } else if (modelReady && resumed) {
                publishLatestFrame(image)
            }
        } finally {
            image.close()
        }
    }

    private fun publishLatestFrame(image: Image) {
        val planes = image.planes
        if (planes.size < 3) return
        val capturedAtNs = SystemClock.elapsedRealtimeNanos()

        // QNN owns no camera buffers. The camera process publishes a compact
        // RGB888 model input; an existing pending frame is overwritten in place.
        synchronized(frameLock) {
            val frame = latestFrame
                ?: if (freeFrames.isEmpty()) RgbFrame() else freeFrames.removeFirst()
            if (frame.copyFrom(image, relativeImageRotation())) {
                frame.capturedAtNs = capturedAtNs
                latestFrame = frame
            } else {
                freeFrames.addLast(frame)
            }
        }
        scheduleInferenceDispatch()
    }

    private fun scheduleInferenceDispatch() {
        if (calibrationActive || accuracyActive || !modelReady ||
            npuService == null || !npuBusy.compareAndSet(false, true)) return
        try {
            inferenceExecutor.execute { dispatchLatestFrame() }
        } catch (_: Exception) {
            npuBusy.set(false)
        }
    }

    private fun dispatchLatestFrame() {
        val frame = synchronized(frameLock) {
            val pending = latestFrame
            if (pending == null) {
                npuBusy.set(false)
                null
            } else {
                latestFrame = null
                pending
            }
        } ?: return

        try {
            val sequence = frameSequence++
            val service = npuService ?: error("NPU service disconnected")
            val mapped = sharedFrameMap ?: error("Shared RGB frame buffer is unavailable")
            inFlightCaptureNs = frame.capturedAtNs
            inFlightDispatchNs = SystemClock.elapsedRealtimeNanos()
            inFlightCameraGeneration = cameraGeneration
            mapped.position(0)
            mapped.put(frame.rgb)
            service.send(Message.obtain(null, ZipDepthService.MSG_PROCESS).apply {
                data = Bundle().apply {
                    putLong(ZipDepthService.KEY_SEQUENCE, sequence)
                }
            })
            // The Binder message is deliberately tiny. The 432 KiB model input
            // stays in shared mapped memory and is not copied through Binder.
            synchronized(frameLock) { freeFrames.addLast(frame) }
        } catch (t: Throwable) {
            synchronized(frameLock) { freeFrames.addLast(frame) }
            npuBusy.set(false)
            modelReady = false
            setStatus("NPU SERVICE SEND FAILED\n${t.message}", true)
        }
    }

    private fun renderPreview(image: Image) {
        val planes = image.planes
        if (planes.size < 3) return
        val slot = previewSlot
        previewSlot = (previewSlot + 1) % previewArgb.size
        val ok = ZipDepthNative.nativePreview(
            planes[0].buffer, planes[1].buffer, planes[2].buffer,
            image.width, image.height,
            planes[0].rowStride, planes[1].rowStride, planes[2].rowStride,
            planes[1].pixelStride, planes[2].pixelStride,
            planes[0].buffer.position(), planes[1].buffer.position(), planes[2].buffer.position(),
            relativeImageRotation(), previewArgb[slot],
        )
        if (!ok) return
        val bitmap = previewBitmaps[slot]
        bitmap.setPixels(previewArgb[slot], 0, ZipDepthNative.OUTPUT_WIDTH, 0, 0, ZipDepthNative.OUTPUT_WIDTH, ZipDepthNative.OUTPUT_HEIGHT)
        runOnUiThread { previewView.setImageBitmap(bitmap) }
    }

    private fun renderDepth(argb: IntArray, res: Int) {
        if (res <= 0 || argb.size < res * res) return
        val bitmaps = depthBitmapCache.getOrPut(res) {
            Array(2) { Bitmap.createBitmap(res, res, Bitmap.Config.ARGB_8888) }
        }
        val slot = depthSlot
        depthSlot = (depthSlot + 1) % bitmaps.size
        val bitmap = bitmaps[slot]
        bitmap.setPixels(argb, 0, res, 0, 0, res, res)
        depthView.setImageBitmap(bitmap)
        if (!depthShown) {
            depthShown = true
            statusView.visibility = View.GONE
        }
    }

    private fun readAndRenderSharedDepth(res: Int) {
        val mapped = sharedFrameMap ?: return
        if (res <= 0 || res * res > remoteDepthArgb.size) return
        val output = mapped.duplicate().order(ByteOrder.nativeOrder())
        output.position(RGB_FRAME_BYTES)
        output.slice().order(ByteOrder.nativeOrder()).asIntBuffer().get(remoteDepthArgb, 0, res * res)
        renderDepth(remoteDepthArgb, res)
    }

    // ---- Synthetic accuracy test ----
    // Runs the ACTIVE model over the bundled testset (fixed 384x384 RGB frames)
    // and scores each raw depth map against the bundled fp32 CPU reference after
    // a per-image affine (scale+shift) fit — the same alignment the downstream
    // calibration applies on-device, so quantization error that an affine fit
    // absorbs is (correctly) not charged to the model.

    private fun startAccuracyTest() {
        if (accuracyActive || !modelReady || npuService == null) return
        val count = try {
            assets.list(TESTSET_DIR).orEmpty().count { it.startsWith("img_") && it.endsWith(".rgb") }
        } catch (_: Throwable) {
            0
        }
        if (count == 0) {
            setStatus("NO BUNDLED TESTSET", true)
            return
        }
        accuracyActive = true
        accuracyIndex = 0
        accuracyCount = count
        accuracyD1.clear()
        accuracyRmse.clear()
        accuracyCorr.clear()
        accuracyButtonView.text = "TESTING 0/$count"
        setStatus("ACCURACY TEST • ${selectedModel?.label}", true)
        // The live dispatch loop is paused via accuracyActive; wait for any
        // in-flight camera inference to drain, then start the sequence.
        Handler(Looper.getMainLooper()).postDelayed({ runNextAccuracyImage() }, 150)
    }

    private fun runNextAccuracyImage() {
        if (!accuracyActive) return
        if (accuracyIndex >= accuracyCount) {
            val d1 = accuracyD1.average()
            val rmse = accuracyRmse.average()
            val corr = accuracyCorr.average()
            val summary = String.format(Locale.US, "d1 %.3f rmse %.4f corr %.3f", d1, rmse, corr)
            selectedModel?.let { option ->
                getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit()
                    .putString(PREF_ACCURACY_PREFIX + option.assetPath, summary)
                    .apply()
            }
            finishAccuracyTest("ACC ${selectedModel?.label}: $summary")
            return
        }
        if (!npuBusy.compareAndSet(false, true)) {
            Handler(Looper.getMainLooper()).postDelayed({ runNextAccuracyImage() }, 60)
            return
        }
        val index = accuracyIndex
        inferenceExecutor.execute {
            try {
                val rgb = assets.open(String.format(Locale.US, "%s/img_%02d.rgb", TESTSET_DIR, index))
                    .use { it.readBytes() }
                accuracyRef = loadReference(index)
                val mapped = sharedFrameMap ?: error("shared buffer missing")
                val service = npuService ?: error("NPU service missing")
                mapped.position(0)
                mapped.put(rgb, 0, RGB_FRAME_BYTES)
                service.send(Message.obtain(null, ZipDepthService.MSG_PROCESS_RAW).apply {
                    data = Bundle().apply { putLong(ZipDepthService.KEY_SEQUENCE, index.toLong()) }
                })
            } catch (t: Throwable) {
                Log.e(TAG, "accuracy image $index failed", t)
                npuBusy.set(false)
                runOnUiThread { finishAccuracyTest("TEST FAILED: ${t.message}") }
            }
        }
    }

    private fun loadReference(index: Int): FloatArray {
        val bytes = assets.open(String.format(Locale.US, "%s/ref_%02d.f32", TESTSET_DIR, index))
            .use { it.readBytes() }
        val floats = FloatArray(bytes.size / 4)
        java.nio.ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        return floats
    }

    // Main thread, after MSG_RESULT_RAW: score prediction vs reference off-thread.
    private fun onAccuracyResult(res: Int) {
        val mapped = sharedFrameMap ?: return
        if (res <= 0 || res * res > remoteDepthRaw.size) return
        val output = mapped.duplicate().order(ByteOrder.nativeOrder())
        output.position(RGB_FRAME_BYTES)
        output.slice().order(ByteOrder.nativeOrder()).asFloatBuffer().get(remoteDepthRaw, 0, res * res)
        val reference = accuracyRef ?: return
        val htpMs = nativeMetrics[2]
        inferenceExecutor.execute {
            val prediction = upsampleBilinear(remoteDepthRaw, res, ZipDepthNative.OUTPUT_WIDTH)
            val scores = affineAlignedScores(prediction, reference)
            runOnUiThread {
                accuracyD1.add(scores[0])
                accuracyRmse.add(scores[1])
                accuracyCorr.add(scores[2])
                accuracyIndex++
                accuracyButtonView.text = "TESTING $accuracyIndex/$accuracyCount"
                setStatus(
                    String.format(
                        Locale.US, "TEST %d/%d  d1 %.3f  htp %.1f ms",
                        accuracyIndex, accuracyCount, scores[0], htpMs,
                    ),
                    true,
                )
                runNextAccuracyImage()
            }
        }
    }

    private fun finishAccuracyTest(message: String) {
        accuracyActive = false
        accuracyRef = null
        accuracyButtonView.text = "TEST ACC"
        setStatus(message, true)
        updateModelSelectorLabel()
        refreshStats(force = true)
        if (modelReady) scheduleInferenceDispatch()
    }

    private fun upsampleBilinear(source: FloatArray, sourceRes: Int, targetRes: Int): FloatArray {
        if (sourceRes == targetRes) return source.copyOf(targetRes * targetRes)
        val target = FloatArray(targetRes * targetRes)
        val scale = sourceRes.toFloat() / targetRes
        for (y in 0 until targetRes) {
            val fy = ((y + 0.5f) * scale - 0.5f).coerceIn(0f, (sourceRes - 1).toFloat())
            val y0 = fy.toInt()
            val y1 = minOf(y0 + 1, sourceRes - 1)
            val ay = fy - y0
            for (x in 0 until targetRes) {
                val fx = ((x + 0.5f) * scale - 0.5f).coerceIn(0f, (sourceRes - 1).toFloat())
                val x0 = fx.toInt()
                val x1 = minOf(x0 + 1, sourceRes - 1)
                val ax = fx - x0
                val top = source[y0 * sourceRes + x0] * (1 - ax) + source[y0 * sourceRes + x1] * ax
                val bottom = source[y1 * sourceRes + x0] * (1 - ax) + source[y1 * sourceRes + x1] * ax
                target[y * targetRes + x] = top * (1 - ay) + bottom * ay
            }
        }
        return target
    }

    /**
     * Least-squares affine fit a*pred+b -> ref, then [d1, rmse, pearson].
     * d1 = fraction of pixels with max(pa/ref, ref/pa) < 1.25 (both clamped
     * positive), the standard depth inlier ratio.
     */
    private fun affineAlignedScores(prediction: FloatArray, reference: FloatArray): DoubleArray {
        val n = minOf(prediction.size, reference.size)
        var sumP = 0.0; var sumR = 0.0; var sumPP = 0.0; var sumPR = 0.0; var sumRR = 0.0
        for (i in 0 until n) {
            val p = prediction[i].toDouble()
            val r = reference[i].toDouble()
            sumP += p; sumR += r; sumPP += p * p; sumPR += p * r; sumRR += r * r
        }
        val denominator = n * sumPP - sumP * sumP
        val a = if (abs(denominator) > 1e-12) (n * sumPR - sumP * sumR) / denominator else 1.0
        val b = (sumR - a * sumP) / n
        var inliers = 0L; var valid = 0L; var squared = 0.0
        for (i in 0 until n) {
            val aligned = a * prediction[i] + b
            val r = reference[i].toDouble()
            if (r <= 1e-6) continue
            valid++
            val pa = maxOf(aligned, 1e-6)
            squared += (pa - r) * (pa - r)
            if (maxOf(pa / r, r / pa) < 1.25) inliers++
        }
        val d1 = if (valid > 0) inliers.toDouble() / valid else 0.0
        val rmse = if (valid > 0) Math.sqrt(squared / valid) else Double.NaN
        val covariance = sumPR / n - (sumP / n) * (sumR / n)
        val varianceP = sumPP / n - (sumP / n) * (sumP / n)
        val varianceR = sumRR / n - (sumR / n) * (sumR / n)
        val pearson = if (varianceP > 1e-12 && varianceR > 1e-12) {
            covariance / Math.sqrt(varianceP * varianceR)
        } else {
            0.0
        }
        return doubleArrayOf(d1, rmse, pearson)
    }

    private fun markCameraFrame() {
        val now = SystemClock.elapsedRealtimeNanos()
        if (cameraRateStartNs == 0L) cameraRateStartNs = now
        cameraRateFrames++
        val elapsed = now - cameraRateStartNs
        if (elapsed >= RATE_WINDOW_NS) {
            val sample = (cameraRateFrames * 1_000_000_000.0 / elapsed).toFloat()
            cameraFps = smoothRate(cameraFps, sample)
            cameraRateStartNs = now
            cameraRateFrames = 0
            runOnUiThread { refreshStats() }
        }
    }

    private fun markDepthFrame(now: Long) {
        if (depthRateStartNs == 0L) depthRateStartNs = now
        depthRateFrames++
        val elapsed = now - depthRateStartNs
        if (elapsed >= RATE_WINDOW_NS) {
            val sample = (depthRateFrames * 1_000_000_000.0 / elapsed).toFloat()
            depthFps = smoothRate(depthFps, sample)
            depthRateStartNs = now
            depthRateFrames = 0
        }
    }

    private fun smoothRate(previous: Float, sample: Float): Float =
        if (previous <= 0f) sample else previous * 0.65f + sample * 0.35f

    private fun resetLiveStats() {
        cameraRateStartNs = 0L
        cameraRateFrames = 0
        cameraFps = 0f
        depthRateStartNs = 0L
        depthRateFrames = 0
        depthFps = 0f
        pipelineLatencyMs = 0f
        dispatchLatencyMs = 0f
        nativeMetrics.fill(0f)
        statsLastDrawNs = 0L
        refreshStats(force = true)
    }

    private fun refreshStats(force: Boolean = false) {
        val now = SystemClock.elapsedRealtimeNanos()
        if (!force && now - statsLastDrawNs < STATS_DRAW_NS) return
        statsLastDrawNs = now
        val waitingMs = if (npuBusy.get() && inFlightDispatchNs > 0L) {
            (now - inFlightDispatchNs) / 1_000_000.0
        } else {
            0.0
        }
        val htpLine = if (calibrationActive) {
            "HTP  PAUSED (CAL)"
        } else if (waitingMs >= 250.0) {
            String.format(Locale.US, "HTP WAIT %.0f ms", waitingMs)
        } else {
            String.format(Locale.US, "HTP  %5.1f ms", nativeMetrics[2])
        }
        val depthLine = if (calibrationActive) {
            "DEPTH PAUSED"
        } else {
            String.format(Locale.US, "DEPTH %5.1f fps", depthFps)
        }
        statsView.text = String.format(
            Locale.US,
            "%s @%dpx\nCAM   %5.1f fps\n%s\n%s\nNATIVE%5.1f ms\nROUND %5.1f ms\nE2E   %5.1f ms",
            selectedModel?.label ?: "MODEL ?",
            modelRes,
            cameraFps,
            depthLine,
            htpLine,
            nativeMetrics[0],
            dispatchLatencyMs,
            pipelineLatencyMs,
        )
        statsView.contentDescription = String.format(
            Locale.US,
            "Camera %.1f fps, depth %.1f fps, HTP %.1f milliseconds, end to end %.1f milliseconds",
            cameraFps,
            depthFps,
            nativeMetrics[2],
            pipelineLatencyMs,
        )
    }

    // Clockwise degrees to rotate the sensor frame upright. Portrait-locked, so the
    // display rotation is 0 on a portrait-native phone and this is SENSOR_ORIENTATION.
    private fun relativeImageRotation(): Int {
        val displayDegrees = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation - displayDegrees + 360) % 360
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
        cameraGeneration++
        cameraOpening.set(false)
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
            setStatus("CAMERA PERMISSION REQUIRED", true)
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()

    private class RgbFrame {
        val rgb = ByteArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 3)
        var capturedAtNs = 0L

        fun copyFrom(image: Image, imageRotation: Int): Boolean {
            val planes = image.planes
            if (planes.size < 3) return false
            return ZipDepthNative.nativePrepareRgb(
                planes[0].buffer, planes[1].buffer, planes[2].buffer,
                image.width, image.height,
                planes[0].rowStride, planes[1].rowStride, planes[2].rowStride,
                planes[1].pixelStride, planes[2].pixelStride,
                planes[0].buffer.position(), planes[1].buffer.position(), planes[2].buffer.position(),
                imageRotation, rgb,
            )
        }
    }

    private data class CameraOption(
        val id: String,
        val facing: Int?,
        val focalLength: Float?,
        val analysisSize: Size?,
    )

    /** One selectable model variant bundled in the APK. */
    private data class ModelOption(
        val assetPath: String,
        val label: String,
    )

    companion object {
        private const val TAG = "ZipDepthDemo"
        private const val CAMERA_PERMISSION = 41
        private const val EXTRA_CAMERA_ID = "camera_id"
        private const val EXTRA_MODEL_ASSET = "model_asset"
        private const val PREFERENCES = "zipdepth_demo"
        private const val PREF_CAMERA_ID = "camera_id"
        private const val PREF_MODEL_ASSET = "model_asset"
        private const val PREF_ACCURACY_PREFIX = "accuracy_"
        private const val PREF_STAGED_PREFIX = "staged_bytes_"
        private const val MODELS_DIR = "models"
        private const val TESTSET_DIR = "testset"
        private const val RATE_WINDOW_NS = 500_000_000L
        private const val STATS_DRAW_NS = 200_000_000L
        private const val RGB_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 3
        private const val DEPTH_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 4
        private const val SHARED_BUFFER_BYTES = RGB_FRAME_BYTES + DEPTH_FRAME_BYTES
    }
}
