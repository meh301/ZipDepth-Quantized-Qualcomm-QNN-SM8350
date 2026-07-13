package org.zipdepth.npudemo

import android.content.Context
import android.media.Image
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.Size
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

data class CalibrationUiState(
    val active: Boolean,
    val acceptedViews: Int,
    val targetViews: Int,
    val message: String,
    val calibration: StoredCameraCalibration? = null,
    val failed: Boolean = false,
)

data class StoredCameraCalibration(
    val cameraId: String,
    val streamWidth: Int,
    val streamHeight: Int,
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val skew: Double,
    val distortion: DoubleArray,
    val rmsPx: Double,
    val sampleCount: Int,
    val timestampMs: Long,
) {
    fun summary(): String = String.format(
        Locale.US,
        "CALIBRATED  RMS %.2f px  %d views",
        rmsPx,
        sampleCount,
    )
}

/**
 * Explicit, low-rate checkerboard calibration path. Camera images are copied
 * before ImageReader releases them and all OpenCV work stays off the camera
 * callback thread. Nothing in this class runs during the normal depth mode.
 */
class CameraCalibrationController(
    context: Context,
    private val onState: (CalibrationUiState) -> Unit,
) : Closeable {
    private val appContext = context.applicationContext
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "CameraCalibration").apply { priority = Thread.NORM_PRIORITY }
    }
    private val processing = AtomicBoolean(false)
    private val stateLock = Any()
    private val samples = ArrayList<CalibrationSample>(TARGET_VIEWS)

    @Volatile private var active = false
    @Volatile private var openCvReady = false
    private var sessionToken = 0
    private var activeCameraId = ""
    private var streamWidth = 0
    private var streamHeight = 0
    private var lastOfferedMs = 0L
    private var lastAcceptedMs = 0L
    private var lastFeedbackMs = 0L

    fun isActive(): Boolean = active

    fun start(cameraId: String, width: Int, height: Int) {
        val token = synchronized(stateLock) {
            sessionToken++
            active = true
            activeCameraId = cameraId
            streamWidth = width
            streamHeight = height
            lastOfferedMs = 0L
            lastAcceptedMs = 0L
            lastFeedbackMs = 0L
            samples.clear()
            sessionToken
        }
        emit(true, 0, "STARTING CALIBRATION ENGINE…")
        executor.execute {
            if (!openCvReady) {
                openCvReady = try {
                    OpenCVLoader.initLocal()
                } catch (t: Throwable) {
                    Log.e(TAG, "OpenCV initialization failed", t)
                    false
                }
            }
            if (!sessionMatches(token)) return@execute
            if (!openCvReady) {
                synchronized(stateLock) { active = false }
                emit(false, 0, "CALIBRATION ENGINE FAILED TO LOAD", failed = true)
            } else {
                emit(
                    true,
                    0,
                    "SHOW THE 10×7-CORNER TARGET\nMOVE IT THROUGH THE WHOLE CAMERA VIEW",
                )
            }
        }
    }

    fun cancel(message: String = "CALIBRATION CANCELLED") {
        val count = synchronized(stateLock) {
            if (!active) return
            active = false
            sessionToken++
            val value = samples.size
            samples.clear()
            value
        }
        emit(false, count, message)
    }

    /** Called while the Image is still owned by MainActivity. */
    fun offer(image: Image) {
        if (!active || !openCvReady) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastOfferedMs < OFFER_INTERVAL_MS || !processing.compareAndSet(false, true)) return
        lastOfferedMs = now

        val token: Int
        val expectedWidth: Int
        val expectedHeight: Int
        synchronized(stateLock) {
            token = sessionToken
            expectedWidth = streamWidth
            expectedHeight = streamHeight
        }
        if (image.width != expectedWidth || image.height != expectedHeight) {
            processing.set(false)
            return
        }
        val gray = copyLuma(image)
        if (gray == null) {
            processing.set(false)
            return
        }
        try {
            executor.execute {
                try {
                    processFrame(gray, expectedWidth, expectedHeight, token)
                } catch (t: Throwable) {
                    Log.e(TAG, "checkerboard processing failed", t)
                } finally {
                    processing.set(false)
                }
            }
        } catch (_: Exception) {
            processing.set(false)
        }
    }

    fun load(cameraId: String, width: Int, height: Int): StoredCameraCalibration? {
        return try {
            val file = calibrationFile(cameraId, width, height)
            if (!file.isFile) return null
            val json = JSONObject(file.readText(Charsets.UTF_8))
            val coefficients = json.getJSONArray("distortionCoefficients")
            StoredCameraCalibration(
                cameraId = json.getString("cameraId"),
                streamWidth = json.getInt("streamWidth"),
                streamHeight = json.getInt("streamHeight"),
                fx = json.getDouble("fx"),
                fy = json.getDouble("fy"),
                cx = json.getDouble("cx"),
                cy = json.getDouble("cy"),
                skew = json.optDouble("skew", 0.0),
                distortion = DoubleArray(coefficients.length()) { coefficients.getDouble(it) },
                rmsPx = json.getDouble("rmsPx"),
                sampleCount = json.getInt("sampleCount"),
                timestampMs = json.getLong("timestampMs"),
            )
        } catch (t: Throwable) {
            Log.w(TAG, "unable to read stored calibration for camera $cameraId", t)
            null
        }
    }

    private fun processFrame(grayBytes: ByteArray, width: Int, height: Int, token: Int) {
        if (!sessionMatches(token)) return
        val gray = Mat(height, width, CvType.CV_8UC1)
        val corners = MatOfPoint2f()
        try {
            gray.put(0, 0, grayBytes)
            val found = Calib3d.findChessboardCornersSB(
                gray,
                PATTERN_SIZE,
                corners,
                Calib3d.CALIB_CB_NORMALIZE_IMAGE or
                    Calib3d.CALIB_CB_EXHAUSTIVE or
                    Calib3d.CALIB_CB_ACCURACY,
            )
            if (!found || corners.total().toInt() != CORNER_COUNT) {
                feedback(
                    token,
                    "BOARD NOT FOUND\nKEEP ALL 11×8 SQUARES VISIBLE AND HOLD STILL",
                )
                return
            }

            val points = corners.toArray()
            val signature = signature(points, width, height)
            if (signature == null) {
                feedback(
                    token,
                    "BOARD FOUND, BUT SIZE IS UNSUITABLE\nMOVE CLOSER OR FARTHER",
                )
                return
            }
            val now = SystemClock.elapsedRealtime()
            val acceptedCount: Int
            val coverage: Coverage
            synchronized(stateLock) {
                if (!active || sessionToken != token) return
                if (now - lastAcceptedMs < MIN_ACCEPT_INTERVAL_MS) return
                if (!isDiverse(signature)) {
                    feedback(
                        token,
                        "BOARD FOUND — VIEW TOO SIMILAR\n${coverage().prompt}",
                    )
                    return
                }
                samples.add(CalibrationSample(points, signature))
                lastAcceptedMs = now
                lastFeedbackMs = now
                acceptedCount = samples.size
                coverage = coverage()
            }

            if (acceptedCount >= TARGET_VIEWS) {
                emit(true, acceptedCount, "CALIBRATING… HOLD STILL")
                finishCalibration(token)
            } else {
                emit(
                    true,
                    acceptedCount,
                    "VIEW ACCEPTED  $acceptedCount/$TARGET_VIEWS\n${coverage.prompt}",
                )
            }
        } finally {
            corners.release()
            gray.release()
        }
    }

    private fun finishCalibration(token: Int) {
        val cameraId: String
        val width: Int
        val height: Int
        val captured: List<CalibrationSample>
        synchronized(stateLock) {
            if (!active || token != sessionToken) return
            cameraId = activeCameraId
            width = streamWidth
            height = streamHeight
            captured = samples.toList()
        }

        try {
            var run = runCalibration(captured, width, height)
            var usedSamples = captured
            val sortedErrors = run.viewErrors.sorted()
            val medianError = sortedErrors[sortedErrors.size / 2]
            val outlierLimit = max(1.25, medianError * 2.5)
            val filtered = captured.filterIndexed { index, _ -> run.viewErrors[index] <= outlierLimit }
            if (filtered.size >= MIN_VIEWS && filtered.size < captured.size) {
                run = runCalibration(filtered, width, height)
                usedSamples = filtered
            }

            val failure = validate(run, width, height, usedSamples.size)
            if (failure != null) {
                synchronized(stateLock) {
                    if (token == sessionToken) active = false
                }
                emit(false, captured.size, "CALIBRATION NOT SAVED\n$failure", failed = true)
                return
            }

            val usedCount = usedSamples.size
            val result = StoredCameraCalibration(
                cameraId = cameraId,
                streamWidth = width,
                streamHeight = height,
                fx = run.cameraMatrix[0],
                fy = run.cameraMatrix[4],
                cx = run.cameraMatrix[2],
                cy = run.cameraMatrix[5],
                skew = run.cameraMatrix[1],
                distortion = run.distortion,
                rmsPx = run.rms,
                sampleCount = usedCount,
                timestampMs = System.currentTimeMillis(),
            )
            if (!sessionMatches(token)) return
            save(result)
            synchronized(stateLock) {
                if (token != sessionToken) return
                active = false
                samples.clear()
            }
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "saved camera %s calibration: %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f rms=%.3f",
                    cameraId,
                    width,
                    height,
                    result.fx,
                    result.fy,
                    result.cx,
                    result.cy,
                    result.rmsPx,
                ),
            )
            emit(false, usedCount, "${result.summary()}\nINTRINSICS SAVED", calibration = result)
        } catch (t: Throwable) {
            Log.e(TAG, "camera calibration failed", t)
            synchronized(stateLock) {
                if (token == sessionToken) active = false
            }
            emit(false, captured.size, "CALIBRATION FAILED\n${t.message ?: "UNKNOWN ERROR"}", failed = true)
        }
    }

    private fun runCalibration(input: List<CalibrationSample>, width: Int, height: Int): CalibrationRun {
        val objectPoints = ArrayList<Mat>(input.size)
        val imagePoints = ArrayList<Mat>(input.size)
        val rvecs = ArrayList<Mat>()
        val tvecs = ArrayList<Mat>()
        val cameraMatrix = Mat.eye(3, 3, CvType.CV_64F)
        val distortion = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)
        try {
            input.forEach { sample ->
                objectPoints.add(MatOfPoint3f(*OBJECT_POINTS))
                imagePoints.add(MatOfPoint2f(*sample.points))
            }
            val rms = Calib3d.calibrateCamera(
                objectPoints,
                imagePoints,
                Size(width.toDouble(), height.toDouble()),
                cameraMatrix,
                distortion,
                rvecs,
                tvecs,
                Calib3d.CALIB_FIX_K4 or Calib3d.CALIB_FIX_K5 or Calib3d.CALIB_FIX_K6,
            )
            val errors = DoubleArray(input.size)
            input.indices.forEach { index ->
                val projected = MatOfPoint2f()
                try {
                    Calib3d.projectPoints(
                        objectPoints[index] as MatOfPoint3f,
                        rvecs[index],
                        tvecs[index],
                        cameraMatrix,
                        distortion,
                        projected,
                    )
                    errors[index] = Core.norm(imagePoints[index], projected, Core.NORM_L2) /
                        sqrt(CORNER_COUNT.toDouble())
                } finally {
                    projected.release()
                }
            }
            val matrixValues = DoubleArray(9)
            cameraMatrix.get(0, 0, matrixValues)
            val distortionValues = DoubleArray(distortion.total().toInt())
            distortion.get(0, 0, distortionValues)
            return CalibrationRun(rms, matrixValues, distortionValues, errors)
        } finally {
            objectPoints.forEach(Mat::release)
            imagePoints.forEach(Mat::release)
            rvecs.forEach(Mat::release)
            tvecs.forEach(Mat::release)
            cameraMatrix.release()
            distortion.release()
        }
    }

    private fun validate(run: CalibrationRun, width: Int, height: Int, views: Int): String? {
        if (views < MIN_VIEWS) return "NOT ENOUGH VALID VIEWS"
        val values = run.cameraMatrix + run.distortion + run.rms
        if (values.any { !it.isFinite() }) return "NON-FINITE CAMERA PARAMETERS"
        val fx = run.cameraMatrix[0]
        val fy = run.cameraMatrix[4]
        val cx = run.cameraMatrix[2]
        val cy = run.cameraMatrix[5]
        val maxDimension = max(width, height).toDouble()
        if (fx !in (maxDimension * 0.2)..(maxDimension * 10.0) ||
            fy !in (maxDimension * 0.2)..(maxDimension * 10.0)
        ) return "IMPLAUSIBLE FOCAL LENGTH"
        if (cx !in (-width * 0.05)..(width * 1.05) || cy !in (-height * 0.05)..(height * 1.05)) {
            return "IMPLAUSIBLE PRINCIPAL POINT"
        }
        if (fx / fy !in 0.5..2.0) return "IMPLAUSIBLE PIXEL ASPECT"
        if (run.distortion.size >= 5 &&
            (abs(run.distortion[0]) > 3.0 || abs(run.distortion[1]) > 10.0 ||
                abs(run.distortion[2]) > 0.5 || abs(run.distortion[3]) > 0.5 ||
                abs(run.distortion[4]) > 100.0)
        ) return "IMPLAUSIBLE DISTORTION COEFFICIENTS"
        if (run.rms > MAX_RMS_PX) {
            return String.format(Locale.US, "REPROJECTION ERROR TOO HIGH (%.2f px)", run.rms)
        }
        return null
    }

    private fun save(calibration: StoredCameraCalibration) {
        val coefficients = JSONArray()
        calibration.distortion.forEach(coefficients::put)
        val json = JSONObject()
            .put("schemaVersion", 1)
            .put("cameraId", calibration.cameraId)
            .put("streamWidth", calibration.streamWidth)
            .put("streamHeight", calibration.streamHeight)
            .put("coordinateSpace", "unrotated_camera_yuv_stream")
            .put("distortionModel", "opencv_brown_conrady_5")
            .put("patternInnerCornersX", PATTERN_COLUMNS)
            .put("patternInnerCornersY", PATTERN_ROWS)
            .put("fx", calibration.fx)
            .put("fy", calibration.fy)
            .put("cx", calibration.cx)
            .put("cy", calibration.cy)
            .put("skew", calibration.skew)
            .put("distortionCoefficients", coefficients)
            .put("rmsPx", calibration.rmsPx)
            .put("sampleCount", calibration.sampleCount)
            .put("timestampMs", calibration.timestampMs)
        val target = calibrationFile(
            calibration.cameraId,
            calibration.streamWidth,
            calibration.streamHeight,
        )
        target.parentFile?.mkdirs()
        val temporary = File(target.parentFile, "${target.name}.tmp")
        temporary.writeText(json.toString(2), Charsets.UTF_8)
        if (target.exists() && !target.delete()) error("Unable to replace old calibration")
        if (!temporary.renameTo(target)) {
            temporary.copyTo(target, overwrite = true)
            temporary.delete()
        }
    }

    private fun calibrationFile(cameraId: String, width: Int, height: Int): File {
        val safeId = cameraId.replace(Regex("[^A-Za-z0-9_.-]"), "_")
        return File(appContext.filesDir, "camera-calibration/${safeId}_${width}x$height.json")
    }

    private fun signature(points: Array<Point>, width: Int, height: Int): ViewSignature? {
        if (points.size != CORNER_COUNT) return null
        val centroidX = points.sumOf(Point::x) / points.size / width
        val centroidY = points.sumOf(Point::y) / points.size / height
        val topLeft = points[0]
        val topRight = points[PATTERN_COLUMNS - 1]
        val bottomLeft = points[(PATTERN_ROWS - 1) * PATTERN_COLUMNS]
        val bottomRight = points.last()
        val area = polygonArea(arrayOf(topLeft, topRight, bottomRight, bottomLeft)) /
            (width.toDouble() * height)
        if (area !in MIN_BOARD_AREA..MAX_BOARD_AREA) return null
        val top = distance(topLeft, topRight)
        val bottom = distance(bottomLeft, bottomRight)
        val left = distance(topLeft, bottomLeft)
        val right = distance(topRight, bottomRight)
        val perspective = abs(ln((top + 1.0) / (bottom + 1.0))) +
            abs(ln((left + 1.0) / (right + 1.0)))
        val angle = atan2(topRight.y - topLeft.y, topRight.x - topLeft.x)
        return ViewSignature(centroidX, centroidY, area, angle, perspective)
    }

    private fun isDiverse(candidate: ViewSignature): Boolean {
        if (samples.isEmpty()) return true
        val closestScore = samples.minOf { sample ->
            val other = sample.signature
            val position = hypot(candidate.x - other.x, candidate.y - other.y) * 2.5
            val scale = abs(ln(candidate.area / other.area)) * 0.45
            val rotation = angleDifference(candidate.angle, other.angle) * 0.8
            val perspective = abs(candidate.perspective - other.perspective) * 0.35
            position + scale + rotation + perspective
        }
        return closestScore >= MIN_DIVERSITY_SCORE
    }

    private fun coverage(): Coverage {
        if (samples.size < 2) return Coverage("MOVE TO A NEW POSITION AND ANGLE")
        val signatures = samples.map(CalibrationSample::signature)
        val xSpan = signatures.maxOf(ViewSignature::x) - signatures.minOf(ViewSignature::x)
        val ySpan = signatures.maxOf(ViewSignature::y) - signatures.minOf(ViewSignature::y)
        val areas = signatures.map(ViewSignature::area)
        val scaleRatio = areas.maxOrNull()!! / areas.minOrNull()!!
        return when {
            xSpan < MIN_POSITION_SPAN -> Coverage("MOVE TARGET FARTHER LEFT AND RIGHT")
            ySpan < MIN_POSITION_SPAN -> Coverage("MOVE TARGET HIGHER AND LOWER")
            scaleRatio < MIN_SCALE_RATIO -> Coverage("ADD BOTH CLOSER AND FARTHER VIEWS")
            else -> Coverage("TILT / ROTATE FOR A NEW VIEW")
        }
    }

    private fun sessionMatches(token: Int): Boolean = synchronized(stateLock) {
        active && token == sessionToken
    }

    private fun feedback(token: Int, message: String) {
        val count = synchronized(stateLock) {
            if (!active || token != sessionToken) return
            val now = SystemClock.elapsedRealtime()
            if (now - lastFeedbackMs < FEEDBACK_INTERVAL_MS) return
            lastFeedbackMs = now
            samples.size
        }
        emit(true, count, message)
    }

    private fun emit(
        isActive: Boolean,
        count: Int,
        message: String,
        calibration: StoredCameraCalibration? = null,
        failed: Boolean = false,
    ) {
        onState(
            CalibrationUiState(
                active = isActive,
                acceptedViews = count,
                targetViews = TARGET_VIEWS,
                message = message,
                calibration = calibration,
                failed = failed,
            ),
        )
    }

    private fun copyLuma(image: Image): ByteArray? {
        val plane = image.planes.firstOrNull() ?: return null
        val buffer = plane.buffer.duplicate()
        val base = buffer.position()
        val limit = buffer.limit()
        val output = ByteArray(image.width * image.height)
        var destination = 0
        for (row in 0 until image.height) {
            val rowStart = base + row * plane.rowStride
            val rowEnd = rowStart + (image.width - 1) * plane.pixelStride
            if (rowStart < 0 || rowEnd >= limit) return null
            if (plane.pixelStride == 1) {
                buffer.position(rowStart)
                buffer.get(output, destination, image.width)
                destination += image.width
            } else {
                for (column in 0 until image.width) {
                    output[destination++] = buffer.get(rowStart + column * plane.pixelStride)
                }
            }
        }
        return output
    }

    override fun close() {
        synchronized(stateLock) {
            active = false
            sessionToken++
            samples.clear()
        }
        executor.shutdownNow()
    }

    private data class CalibrationSample(val points: Array<Point>, val signature: ViewSignature)
    private data class ViewSignature(
        val x: Double,
        val y: Double,
        val area: Double,
        val angle: Double,
        val perspective: Double,
    )
    private data class Coverage(val prompt: String)
    private data class CalibrationRun(
        val rms: Double,
        val cameraMatrix: DoubleArray,
        val distortion: DoubleArray,
        val viewErrors: DoubleArray,
    )

    companion object {
        private const val TAG = "CameraCalibration"
        private const val PATTERN_COLUMNS = 10
        private const val PATTERN_ROWS = 7
        private const val CORNER_COUNT = PATTERN_COLUMNS * PATTERN_ROWS
        private const val MIN_VIEWS = 10
        private const val TARGET_VIEWS = 12
        private const val OFFER_INTERVAL_MS = 250L
        private const val MIN_ACCEPT_INTERVAL_MS = 400L
        private const val FEEDBACK_INTERVAL_MS = 700L
        private const val MIN_BOARD_AREA = 0.012
        private const val MAX_BOARD_AREA = 0.72
        private const val MIN_DIVERSITY_SCORE = 0.035
        private const val MIN_POSITION_SPAN = 0.18
        private const val MIN_SCALE_RATIO = 1.3
        private const val MAX_RMS_PX = 1.5
        private val PATTERN_SIZE = Size(PATTERN_COLUMNS.toDouble(), PATTERN_ROWS.toDouble())
        private val OBJECT_POINTS = Array(CORNER_COUNT) { index ->
            Point3(
                (index % PATTERN_COLUMNS).toDouble(),
                (index / PATTERN_COLUMNS).toDouble(),
                0.0,
            )
        }

        private fun distance(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)

        private fun polygonArea(points: Array<Point>): Double {
            var twiceArea = 0.0
            points.indices.forEach { index ->
                val next = points[(index + 1) % points.size]
                twiceArea += points[index].x * next.y - next.x * points[index].y
            }
            return abs(twiceArea) * 0.5
        }

        private fun angleDifference(a: Double, b: Double): Double {
            var difference = abs(a - b) % Math.PI
            if (difference > Math.PI / 2.0) difference = Math.PI - difference
            return difference
        }
    }
}
