package org.zipdepth.npudemo

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Owns ORT/QNN in a process that never opens or references the camera. */
class ZipDepthService : Service() {
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "ZipDepthRemoteNpu").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val initialized = AtomicBoolean(false)
    // Sized for the largest variant (384); smaller models fill a res*res prefix.
    private val outputArgb = IntArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)
    private val outputDepth = FloatArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)
    private val inputRgb = ByteArray(RGB_FRAME_BYTES)
    private val metrics = FloatArray(6)
    @Volatile private var modelRes = 0
    private var sharedFrameMap: MappedByteBuffer? = null
    @Volatile private var client: Messenger? = null

    private val incoming = Messenger(Handler(Looper.getMainLooper()) { message ->
        when (message.what) {
            MSG_INITIALIZE -> initialize(message)
            MSG_PROCESS -> process(message, raw = false)
            MSG_PROCESS_RAW -> process(message, raw = true)
            MSG_RESTART -> restartProcess()
        }
        true
    })

    override fun onBind(intent: Intent?): IBinder = incoming.binder

    private fun initialize(message: Message) {
        client = message.replyTo
        val modelPath = message.data.getString(KEY_MODEL_PATH).orEmpty()
        val framePath = message.data.getString(KEY_FRAME_PATH).orEmpty()
        executor.execute {
            val result = try {
                val frameFile = File(framePath)
                if (!frameFile.isFile || frameFile.length() != SHARED_BUFFER_BYTES.toLong()) {
                    error("invalid shared frame buffer '$framePath'")
                }
                if (sharedFrameMap == null) {
                    RandomAccessFile(frameFile, "rw").use { randomFile ->
                        sharedFrameMap = randomFile.channel.map(
                            FileChannel.MapMode.READ_WRITE,
                            0L,
                            SHARED_BUFFER_BYTES.toLong(),
                        )
                    }
                }
                // Model switch: tear down the previous session first. The native
                // layer keeps libonnxruntime loaded, so this is a clean re-init.
                if (initialized.getAndSet(false)) {
                    ZipDepthNative.nativeShutdown()
                }
                ZipDepthNative.nativeInit(modelPath)
            } catch (t: Throwable) {
                "ERROR: ${t.javaClass.simpleName}: ${t.message ?: "native init threw"}"
            }
            val ok = !result.startsWith("ERROR")
            modelRes = if (ok) ZipDepthNative.nativeModelInputSize() else 0
            initialized.set(ok)
            Log.i(TAG, "remote model init: ready=$ok res=$modelRes result=\"$result\"")
            send(MSG_INITIALIZED, Bundle().apply {
                putString(KEY_RESULT, result)
                putInt(KEY_MODEL_RES, modelRes)
            })
        }
    }

    private fun process(message: Message, raw: Boolean) {
        val sequence = message.data.getLong(KEY_SEQUENCE)
        val mapped = sharedFrameMap
        if (!initialized.get() || mapped == null) {
            sendResult(sequence, false, "remote NPU is not initialized", raw)
            return
        }
        executor.execute {
            val ok = try {
                mapped.position(0)
                mapped.get(inputRgb)
                val res = modelRes
                if (raw) {
                    val processed = ZipDepthNative.nativeProcessRgbRaw(inputRgb, outputDepth, metrics)
                    if (processed) {
                        val output = mapped.duplicate().order(ByteOrder.nativeOrder())
                        output.position(RGB_FRAME_BYTES)
                        output.slice().order(ByteOrder.nativeOrder()).asFloatBuffer()
                            .put(outputDepth, 0, res * res)
                    }
                    processed
                } else {
                    val processed = ZipDepthNative.nativeProcessRgb(inputRgb, outputArgb, metrics)
                    if (processed) {
                        val output = mapped.duplicate().order(ByteOrder.nativeOrder())
                        output.position(RGB_FRAME_BYTES)
                        output.slice().order(ByteOrder.nativeOrder()).asIntBuffer()
                            .put(outputArgb, 0, res * res)
                    }
                    processed
                }
            } catch (t: Throwable) {
                Log.e(TAG, "remote inference threw", t)
                false
            }
            sendResult(sequence, ok, if (ok) "" else "QNN inference failed; see ZipDepthDemo log", raw)
        }
    }

    // Last-resort recovery requested by the client after a failed model switch:
    // die and let AMS give the next bind a pristine process (fresh QNN state).
    private fun restartProcess() {
        Log.w(TAG, "client requested NPU process restart")
        executor.execute {
            try {
                ZipDepthNative.nativeShutdown()
            } finally {
                Process.killProcess(Process.myPid())
            }
        }
    }

    private fun sendResult(sequence: Long, ok: Boolean, error: String, raw: Boolean) {
        send(if (raw) MSG_RESULT_RAW else MSG_RESULT, Bundle().apply {
            putLong(KEY_SEQUENCE, sequence)
            putBoolean(KEY_OK, ok)
            putString(KEY_ERROR, error)
            putInt(KEY_MODEL_RES, modelRes)
            if (ok) {
                putFloatArray(KEY_METRICS, metrics)
            }
        })
    }

    private fun send(what: Int, data: Bundle) {
        try {
            client?.send(Message.obtain(null, what).apply { this.data = data })
        } catch (t: Throwable) {
            Log.e(TAG, "unable to return remote NPU message $what", t)
        }
    }

    override fun onDestroy() {
        initialized.set(false)
        executor.execute { ZipDepthNative.nativeShutdown() }
        executor.shutdown()
        super.onDestroy()
    }

    companion object {
        const val MSG_INITIALIZE = 1
        const val MSG_INITIALIZED = 2
        const val MSG_PROCESS = 3
        const val MSG_RESULT = 4
        const val MSG_PROCESS_RAW = 5
        const val MSG_RESULT_RAW = 6
        const val MSG_RESTART = 7
        const val KEY_MODEL_PATH = "model_path"
        const val KEY_FRAME_PATH = "frame_path"
        const val KEY_RESULT = "result"
        const val KEY_SEQUENCE = "sequence"
        const val KEY_OK = "ok"
        const val KEY_ERROR = "error"
        const val KEY_METRICS = "metrics"
        const val KEY_MODEL_RES = "model_res"
        private const val RGB_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 3
        private const val DEPTH_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 4
        private const val SHARED_BUFFER_BYTES = RGB_FRAME_BYTES + DEPTH_FRAME_BYTES
        private const val TAG = "ZipDepthDemo"
    }
}
