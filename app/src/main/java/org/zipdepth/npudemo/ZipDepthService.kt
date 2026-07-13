package org.zipdepth.npudemo

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
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
    private val outputArgb = IntArray(ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT)
    private val inputRgb = ByteArray(RGB_FRAME_BYTES)
    private val metrics = FloatArray(6)
    private var sharedFrameMap: MappedByteBuffer? = null
    @Volatile private var client: Messenger? = null

    private val incoming = Messenger(Handler(Looper.getMainLooper()) { message ->
        when (message.what) {
            MSG_INITIALIZE -> initialize(message)
            MSG_PROCESS -> process(message)
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
                RandomAccessFile(frameFile, "rw").use { randomFile ->
                    sharedFrameMap = randomFile.channel.map(
                        FileChannel.MapMode.READ_WRITE,
                        0L,
                        SHARED_BUFFER_BYTES.toLong(),
                    )
                }
                ZipDepthNative.nativeInit(modelPath)
            } catch (t: Throwable) {
                "ERROR: ${t.javaClass.simpleName}: ${t.message ?: "native init threw"}"
            }
            initialized.set(!result.startsWith("ERROR"))
            Log.i(TAG, "remote model init: ready=${initialized.get()} result=\"$result\"")
            send(MSG_INITIALIZED, Bundle().apply { putString(KEY_RESULT, result) })
        }
    }

    private fun process(message: Message) {
        val sequence = message.data.getLong(KEY_SEQUENCE)
        val mapped = sharedFrameMap
        if (!initialized.get() || mapped == null) {
            sendResult(sequence, false, "remote NPU is not initialized")
            return
        }
        executor.execute {
            val ok = try {
                mapped.position(0)
                mapped.get(inputRgb)
                val processed = ZipDepthNative.nativeProcessRgb(inputRgb, outputArgb, metrics)
                if (processed) {
                    val output = mapped.duplicate().order(ByteOrder.nativeOrder())
                    output.position(RGB_FRAME_BYTES)
                    output.slice().order(ByteOrder.nativeOrder()).asIntBuffer().put(outputArgb)
                }
                processed
            } catch (t: Throwable) {
                Log.e(TAG, "remote inference threw", t)
                false
            }
            sendResult(sequence, ok, if (ok) "" else "QNN inference failed; see ZipDepthDemo log")
        }
    }

    private fun sendResult(sequence: Long, ok: Boolean, error: String) {
        send(MSG_RESULT, Bundle().apply {
            putLong(KEY_SEQUENCE, sequence)
            putBoolean(KEY_OK, ok)
            putString(KEY_ERROR, error)
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
        const val KEY_MODEL_PATH = "model_path"
        const val KEY_FRAME_PATH = "frame_path"
        const val KEY_RESULT = "result"
        const val KEY_SEQUENCE = "sequence"
        const val KEY_OK = "ok"
        const val KEY_ERROR = "error"
        const val KEY_METRICS = "metrics"
        private const val RGB_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 3
        private const val DEPTH_FRAME_BYTES =
            ZipDepthNative.OUTPUT_WIDTH * ZipDepthNative.OUTPUT_HEIGHT * 4
        private const val SHARED_BUFFER_BYTES = RGB_FRAME_BYTES + DEPTH_FRAME_BYTES
        private const val TAG = "ZipDepthDemo"
    }
}
