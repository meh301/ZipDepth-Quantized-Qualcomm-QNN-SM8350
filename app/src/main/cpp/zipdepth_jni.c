#define _GNU_SOURCE

#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>
#include <time.h>

#include "ort/onnxruntime_c_api.h"

#define TAG "ZipDepthDemo"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MODEL_W 384
#define MODEL_H 384
#define MODEL_PIXELS (MODEL_W * MODEL_H)
#define SAMPLE_COUNT 4096

typedef struct {
    void *ort_library;
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    OrtRunOptions *run_options;
    char input_name[128];
    char output_name[128];
    char backend_name[96];
    char last_error[512];
    float input[3 * MODEL_PIXELS];
    float percentile_samples[SAMPLE_COUNT];
    jint output_pixels[MODEL_PIXELS];
    float low_ema;
    float high_ema;
    int range_ready;
    int ready;
} ZipDepthState;

static ZipDepthState Z;

static double milliseconds(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1000000.0;
}

static void set_error(const char *where, const char *detail) {
    snprintf(Z.last_error, sizeof(Z.last_error), "%s: %s", where,
             detail ? detail : "unknown error");
    LOGE("%s", Z.last_error);
}

static int ort_ok(OrtStatus *status, const char *where) {
    if (!status) return 1;
    const char *message = Z.api ? Z.api->GetErrorMessage(status) : "ORT error";
    set_error(where, message);
    if (Z.api) Z.api->ReleaseStatus(status);
    return 0;
}

static void native_library_directory(char *output, size_t output_size) {
    output[0] = '\0';
    Dl_info info;
    if (dladdr((void *)native_library_directory, &info) && info.dli_fname) {
        const char *separator = strrchr(info.dli_fname, '/');
        if (separator) {
            snprintf(output, output_size, "%.*s",
                     (int)(separator - info.dli_fname), info.dli_fname);
        }
    }
}

static void configure_fastrpc_paths(const char *directory) {
    const char *old_ld = getenv("LD_LIBRARY_PATH");
    const char *old_adsp = getenv("ADSP_LIBRARY_PATH");
    char path[1600];
    snprintf(path, sizeof(path), "%s%s%s", directory,
             old_ld && old_ld[0] ? ":" : "", old_ld && old_ld[0] ? old_ld : "");
    setenv("LD_LIBRARY_PATH", path, 1);
    snprintf(path, sizeof(path), "%s%s%s", directory,
             old_adsp && old_adsp[0] ? ";" : "",
             old_adsp && old_adsp[0] ? old_adsp : "");
    setenv("ADSP_LIBRARY_PATH", path, 1);
    LOGI("QNN runtime directory: %s", directory);
}

static int open_qnn_session(const char *model_path, const char *backend_path,
                            int strict) {
    OrtSessionOptions *options = NULL;
    if (!ort_ok(Z.api->CreateSessionOptions(&options), "CreateSessionOptions")) return 0;

    ort_ok(Z.api->SetSessionLogSeverityLevel(options, 2), "SetSessionLogSeverityLevel");
    ort_ok(Z.api->SetIntraOpNumThreads(options, 1), "SetIntraOpNumThreads");
    ort_ok(Z.api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL),
           "SetSessionGraphOptimizationLevel");

    // NOTE: htp_performance_mode and rpc_control_latency are set PER-RUN (via the
    // run options below), NOT here. Setting the HTP performance mode both as a
    // provider option and per-run makes QNN reject the Run with "there is already
    // a pending HTP performance mode config". Only qnn_context_priority (which has
    // no per-run equivalent) is a provider option.
    const char *keys[] = {
        "backend_path",
        "offload_graph_io_quantization",
        "qnn_context_priority",
    };
    const char *values[] = {
        backend_path,
        "0",
        "high",
    };
    OrtStatus *append_status = Z.api->SessionOptionsAppendExecutionProvider(
        options, "QNN", keys, values, 3);
    if (append_status) {
        ort_ok(append_status, "Append QNN execution provider");
        Z.api->ReleaseSessionOptions(options);
        return 0;
    }
    if (strict) {
        ort_ok(Z.api->AddSessionConfigEntry(
                   options, "session.disable_cpu_ep_fallback", "1"),
               "Disable CPU fallback");
    }

    OrtStatus *session_status = Z.api->CreateSession(
        Z.env, model_path, options, &Z.session);
    Z.api->ReleaseSessionOptions(options);
    if (session_status) {
        ort_ok(session_status, strict ? "Open strict QNN EPContext" : "Open QNN EPContext");
        Z.session = NULL;
        return 0;
    }
    snprintf(Z.backend_name, sizeof(Z.backend_name), "%s",
             strict ? "QNN HTP • EPContext • all nodes"
                    : "QNN HTP • EPContext • CPU I/O boundary");
    return 1;
}

// Run one inference on the zero-initialized input so the expensive first-call
// HTP graph finalization happens here (reported to the user as "initializing")
// rather than stalling the first live camera frame. It also surfaces any
// inference-time failure at init, where logcat already reaches. Returns 1 on
// success; 0 (with a logged reason) if the warmup run itself fails.
static int warmup_inference(void) {
    // One inference on the zero input so the expensive first-call HTP graph
    // finalization happens here instead of on the first live frame.
    LOGI("warmup: running the first OrtRun…");
    double t0 = milliseconds();
    const int64_t shape[] = {1, 3, MODEL_H, MODEL_W};
    OrtValue *input_tensor = NULL;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.memory_info, Z.input, sizeof(Z.input), shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
                "warmup CreateTensor")) return 0;
    const char *input_names[] = {Z.input_name};
    const char *output_names[] = {Z.output_name};
    OrtValue *output_tensor = NULL;
    OrtStatus *status = Z.api->Run(Z.session, Z.run_options, input_names,
                                   (const OrtValue *const *)&input_tensor, 1,
                                   output_names, 1, &output_tensor);
    Z.api->ReleaseValue(input_tensor);
    if (!ort_ok(status, "warmup OrtRun")) return 0;
    if (output_tensor) Z.api->ReleaseValue(output_tensor);
    LOGI("warmup: first OrtRun completed in %.1f ms", milliseconds() - t0);
    return 1;
}

static int initialize_ort(const char *model_path) {
    char board[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.board.platform", board);
    if (!strstr(board, "lahaina") && !strstr(board, "sm8350")) {
        char reason[160];
        snprintf(reason, sizeof(reason),
                 "this context targets SM8350/V68; detected board '%s'", board);
        set_error("Unsupported SoC", reason);
        return 0;
    }

    char library_directory[512];
    native_library_directory(library_directory, sizeof(library_directory));
    if (!library_directory[0]) {
        set_error("Runtime setup", "could not locate native library directory");
        return 0;
    }
    configure_fastrpc_paths(library_directory);

    Z.ort_library = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!Z.ort_library) {
        set_error("dlopen libonnxruntime.so", dlerror());
        return 0;
    }
    const OrtApiBase *(*get_api_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(Z.ort_library, "OrtGetApiBase");
    if (!get_api_base) {
        set_error("dlsym OrtGetApiBase", dlerror());
        return 0;
    }
    Z.api = get_api_base()->GetApi(ORT_API_VERSION);
    if (!Z.api) {
        set_error("GetApi", "requested ORT C API is unavailable");
        return 0;
    }
    if (!ort_ok(Z.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "zipdepth-rgb-demo", &Z.env),
                "CreateEnv")) return 0;

    char backend_path[640];
    snprintf(backend_path, sizeof(backend_path), "%s/libQnnHtp.so", library_directory);
    if (!open_qnn_session(model_path, backend_path, 1)) {
        LOGI("Strict session rejected; retrying with CPU boundary nodes allowed");
        if (!open_qnn_session(model_path, backend_path, 0)) return 0;
    }

    if (!ort_ok(Z.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                            &Z.memory_info),
                "CreateCpuMemoryInfo")) return 0;

    OrtAllocator *allocator = NULL;
    if (!ort_ok(Z.api->GetAllocatorWithDefaultOptions(&allocator),
                "GetAllocatorWithDefaultOptions")) return 0;
    char *name = NULL;
    if (!ort_ok(Z.api->SessionGetInputName(Z.session, 0, allocator, &name),
                "SessionGetInputName")) return 0;
    snprintf(Z.input_name, sizeof(Z.input_name), "%s", name);
    allocator->Free(allocator, name);
    name = NULL;
    if (!ort_ok(Z.api->SessionGetOutputName(Z.session, 0, allocator, &name),
                "SessionGetOutputName")) return 0;
    snprintf(Z.output_name, sizeof(Z.output_name), "%s", name);
    allocator->Free(allocator, name);

    // No HTP burst/power-vote options: this app runs in an UNSIGNED protection
    // domain, where the HTP setPowerConfig (HAP_Power DCVS) call is not permitted
    // and fails the Run. We leave the DSP at its default DCVS. Kotlin feeds it
    // through a bounded latest-frame mailbox in a camera-free Android service
    // process, while the camera remains a continuous independent producer.
    if (!ort_ok(Z.api->CreateRunOptions(&Z.run_options), "CreateRunOptions")) return 0;

    LOGI("ZipDepth session opened (board=%s backend='%s' input=%s output=%s)",
         board, Z.backend_name, Z.input_name, Z.output_name);
    if (!warmup_inference()) {
        // The graph loaded but the first inference could not run — report the
        // specific ORT/QNN reason instead of claiming the NPU is ready.
        if (!Z.last_error[0]) set_error("Warmup inference", "first OrtRun failed");
        return 0;
    }
    Z.ready = 1;
    LOGI("ZipDepth ready: board=%s backend='%s' input=%s output=%s",
         board, Z.backend_name, Z.input_name, Z.output_name);
    return 1;
}

static inline float clampf(float value, float low, float high) {
    return value < low ? low : value > high ? high : value;
}

static inline uint8_t plane_value(const uint8_t *plane, int offset,
                                  int row_stride, int pixel_stride,
                                  int x, int y) {
    return plane[offset + y * row_stride + x * pixel_stride];
}

static float sample_luma(const uint8_t *y_plane, int offset, int row_stride,
                         int width, int height, float x, float y) {
    x = clampf(x, 0.0f, (float)(width - 1));
    y = clampf(y, 0.0f, (float)(height - 1));
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1 < width ? x0 + 1 : x0;
    int y1 = y0 + 1 < height ? y0 + 1 : y0;
    float ax = x - x0;
    float ay = y - y0;
    float top = plane_value(y_plane, offset, row_stride, 1, x0, y0) * (1.0f - ax) +
                plane_value(y_plane, offset, row_stride, 1, x1, y0) * ax;
    float bottom = plane_value(y_plane, offset, row_stride, 1, x0, y1) * (1.0f - ax) +
                   plane_value(y_plane, offset, row_stride, 1, x1, y1) * ax;
    return top * (1.0f - ay) + bottom * ay;
}

static void oriented_to_sensor(float oriented_x, float oriented_y,
                               int width, int height, int rotation,
                               float *sensor_x, float *sensor_y) {
    switch (rotation) {
        case 90:
            *sensor_x = oriented_y;
            *sensor_y = (float)(height - 1) - oriented_x;
            break;
        case 180:
            *sensor_x = (float)(width - 1) - oriented_x;
            *sensor_y = (float)(height - 1) - oriented_y;
            break;
        case 270:
            *sensor_x = (float)(width - 1) - oriented_y;
            *sensor_y = oriented_x;
            break;
        default:
            *sensor_x = oriented_x;
            *sensor_y = oriented_y;
            break;
    }
}

// Center-crop geometry for mapping the MODEL_WxMODEL_H square onto the upright
// (display-oriented) camera frame. Shared by the model preprocessing and the
// preview so the depth map and the RGB image show pixel-identical framing.
typedef struct {
    int width;
    int height;
    int rotation;
    float crop_x;
    float crop_y;
    float scale;
} UprightMap;

static UprightMap upright_map(int width, int height, int rotation) {
    int oriented_width = (rotation == 90 || rotation == 270) ? height : width;
    int oriented_height = (rotation == 90 || rotation == 270) ? width : height;
    int crop = oriented_width < oriented_height ? oriented_width : oriented_height;
    UprightMap map;
    map.width = width;
    map.height = height;
    map.rotation = rotation;
    map.crop_x = (oriented_width - crop) * 0.5f;
    map.crop_y = (oriented_height - crop) * 0.5f;
    map.scale = (float)crop / MODEL_W;
    return map;
}

// Sample the upright, center-cropped RGB for one model pixel. Outputs are in
// [0, 255]. YUV_420_888 handling per the Android contract: Y pixel-stride is 1,
// chroma is read with its own row/pixel stride at half resolution.
static void sample_upright_rgb(const uint8_t *y_plane, const uint8_t *u_plane,
                               const uint8_t *v_plane, const UprightMap *map,
                               int y_row_stride, int u_row_stride, int v_row_stride,
                               int u_pixel_stride, int v_pixel_stride,
                               int y_offset, int u_offset, int v_offset,
                               int model_x, int model_y,
                               float *r, float *g, float *b) {
    int width = map->width;
    int height = map->height;
    float oriented_x = map->crop_x + (model_x + 0.5f) * map->scale - 0.5f;
    float oriented_y = map->crop_y + (model_y + 0.5f) * map->scale - 0.5f;
    float sensor_x, sensor_y;
    oriented_to_sensor(oriented_x, oriented_y, width, height, map->rotation,
                       &sensor_x, &sensor_y);
    sensor_x = clampf(sensor_x, 0.0f, (float)(width - 1));
    sensor_y = clampf(sensor_y, 0.0f, (float)(height - 1));

    float y_value = sample_luma(y_plane, y_offset, y_row_stride,
                                width, height, sensor_x, sensor_y);
    int chroma_x = ((int)(sensor_x + 0.5f) / 2);
    int chroma_y = ((int)(sensor_y + 0.5f) / 2);
    int chroma_width = (width + 1) / 2;
    int chroma_height = (height + 1) / 2;
    if (chroma_x >= chroma_width) chroma_x = chroma_width - 1;
    if (chroma_y >= chroma_height) chroma_y = chroma_height - 1;
    float u_value = plane_value(u_plane, u_offset, u_row_stride,
                                u_pixel_stride, chroma_x, chroma_y) - 128.0f;
    float v_value = plane_value(v_plane, v_offset, v_row_stride,
                                v_pixel_stride, chroma_x, chroma_y) - 128.0f;

    // Camera2 YUV_420_888 convention: limited-range BT.601 is the most
    // interoperable approximation for real-time preview frames.
    float luminance = 1.164383f * (y_value - 16.0f);
    if (luminance < 0.0f) luminance = 0.0f;
    *r = clampf(luminance + 1.596027f * v_value, 0.0f, 255.0f);
    *g = clampf(luminance - 0.391762f * u_value - 0.812968f * v_value, 0.0f, 255.0f);
    *b = clampf(luminance + 2.017232f * u_value, 0.0f, 255.0f);
}

static void preprocess_yuv(const uint8_t *y_plane, const uint8_t *u_plane,
                           const uint8_t *v_plane, int width, int height,
                           int y_row_stride, int u_row_stride, int v_row_stride,
                           int u_pixel_stride, int v_pixel_stride,
                           int y_offset, int u_offset, int v_offset,
                           int rotation) {
    UprightMap map = upright_map(width, height, rotation);
    float *red = Z.input;
    float *green = Z.input + MODEL_PIXELS;
    float *blue = Z.input + 2 * MODEL_PIXELS;
    for (int model_y = 0; model_y < MODEL_H; ++model_y) {
        for (int model_x = 0; model_x < MODEL_W; ++model_x) {
            float r, g, b;
            sample_upright_rgb(y_plane, u_plane, v_plane, &map,
                               y_row_stride, u_row_stride, v_row_stride,
                               u_pixel_stride, v_pixel_stride,
                               y_offset, u_offset, v_offset,
                               model_x, model_y, &r, &g, &b);
            int index = model_y * MODEL_W + model_x;
            red[index] = r * (1.0f / 255.0f);
            green[index] = g * (1.0f / 255.0f);
            blue[index] = b * (1.0f / 255.0f);
        }
    }
}

static void preprocess_rgb888(const uint8_t *rgb) {
    float *red = Z.input;
    float *green = Z.input + MODEL_PIXELS;
    float *blue = Z.input + 2 * MODEL_PIXELS;
    const float scale = 1.0f / 255.0f;
    for (int index = 0; index < MODEL_PIXELS; ++index) {
        red[index] = rgb[index * 3] * scale;
        green[index] = rgb[index * 3 + 1] * scale;
        blue[index] = rgb[index * 3 + 2] * scale;
    }
}

// Preview scratch is written only from the camera thread (single caller), kept
// separate from Z.output_pixels so preview and NPU inference never share state.
static jint preview_pixels[MODEL_PIXELS];

static int compare_floats(const void *left, const void *right) {
    float a = *(const float *)left;
    float b = *(const float *)right;
    return (a > b) - (a < b);
}

static int estimate_range(const float *depth, float *low, float *high) {
    int count = 0;
    const int step = MODEL_PIXELS / SAMPLE_COUNT;
    for (int index = 0; index < MODEL_PIXELS && count < SAMPLE_COUNT; index += step) {
        float value = depth[index];
        if (isfinite(value)) Z.percentile_samples[count++] = value;
    }
    if (count < 32) return 0;
    qsort(Z.percentile_samples, (size_t)count, sizeof(float), compare_floats);
    float frame_low = Z.percentile_samples[(int)(count * 0.02f)];
    float frame_high = Z.percentile_samples[(int)(count * 0.98f)];
    if (!(frame_high > frame_low + 1e-7f)) return 0;

    if (!Z.range_ready) {
        Z.low_ema = frame_low;
        Z.high_ema = frame_high;
        Z.range_ready = 1;
    } else {
        const float alpha = 0.18f;
        Z.low_ema += alpha * (frame_low - Z.low_ema);
        Z.high_ema += alpha * (frame_high - Z.high_ema);
    }
    *low = Z.low_ema;
    *high = Z.high_ema;
    return 1;
}

static inline int color_channel(float value) {
    value = clampf(value, 0.0f, 1.0f);
    return (int)(value * 255.0f + 0.5f);
}

static uint32_t turbo(float x) {
    x = clampf(x, 0.0f, 1.0f);
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x3 * x;
    float x5 = x4 * x;
    float r = 0.13572138f + 4.61539260f * x - 42.66032258f * x2 +
              132.13108234f * x3 - 152.94239396f * x4 + 59.28637943f * x5;
    float g = 0.09140261f + 2.19418839f * x + 4.84296658f * x2 -
              14.18503333f * x3 + 4.27729857f * x4 + 2.82956604f * x5;
    float b = 0.10667330f + 12.64194608f * x - 60.58204836f * x2 +
              110.36276771f * x3 - 89.90310912f * x4 + 27.34824973f * x5;
    return 0xFF000000u | ((uint32_t)color_channel(r) << 16) |
           ((uint32_t)color_channel(g) << 8) | (uint32_t)color_channel(b);
}

static jboolean run_preprocessed(JNIEnv *env, jintArray output_argb,
                                 jfloatArray metrics, double start,
                                 double after_preprocess, int frame, int trace) {
    if (trace) LOGI("nativeProcess[%d]: preprocessed in %.1f ms; calling OrtRun",
                    frame, after_preprocess - start);

    const int64_t shape[] = {1, 3, MODEL_H, MODEL_W};
    OrtValue *input_tensor = NULL;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.memory_info, Z.input, sizeof(Z.input), shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
                "Create input tensor")) return JNI_FALSE;
    const char *input_names[] = {Z.input_name};
    const char *output_names[] = {Z.output_name};
    OrtValue *output_tensor = NULL;
    OrtStatus *run_status = Z.api->Run(
        Z.session, Z.run_options, input_names, (const OrtValue *const *)&input_tensor, 1,
        output_names, 1, &output_tensor);
    Z.api->ReleaseValue(input_tensor);
    if (!ort_ok(run_status, "QNN inference")) return JNI_FALSE;
    double after_inference = milliseconds();
    if (trace) LOGI("nativeProcess[%d]: OrtRun ok in %.1f ms",
                    frame, after_inference - after_preprocess);

    float *depth = NULL;
    if (!ort_ok(Z.api->GetTensorMutableData(output_tensor, (void **)&depth),
                "Read depth output") || !depth) {
        if (output_tensor) Z.api->ReleaseValue(output_tensor);
        return JNI_FALSE;
    }
    float low, high;
    if (!estimate_range(depth, &low, &high)) {
        if (trace) LOGE("nativeProcess[%d]: estimate_range failed (degenerate depth)", frame);
        Z.api->ReleaseValue(output_tensor);
        return JNI_FALSE;
    }
    float inverse_span = 1.0f / (high - low);
    for (int index = 0; index < MODEL_PIXELS; ++index) {
        float value = isfinite(depth[index]) ? depth[index] : high;
        float near_value = 1.0f - (value - low) * inverse_span;
        Z.output_pixels[index] = (jint)turbo(near_value);
    }
    Z.api->ReleaseValue(output_tensor);
    (*env)->SetIntArrayRegion(env, output_argb, 0, MODEL_PIXELS, Z.output_pixels);
    double after_postprocess = milliseconds();

    jfloat values[6] = {
        (jfloat)(after_postprocess - start),
        (jfloat)(after_preprocess - start),
        (jfloat)(after_inference - after_preprocess),
        (jfloat)(after_postprocess - after_inference),
        low,
        high,
    };
    (*env)->SetFloatArrayRegion(env, metrics, 0, 6, values);
    if (trace) LOGI("nativeProcess[%d]: done, total %.1f ms, depth window [%.4f .. %.4f]",
                    frame, after_postprocess - start, low, high);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativeInit(
    JNIEnv *env, jobject ignored, jstring model_path_string) {
    (void)ignored;
    if (Z.ready) return (*env)->NewStringUTF(env, Z.backend_name);
    const char *model_path = (*env)->GetStringUTFChars(env, model_path_string, NULL);
    int ok = initialize_ort(model_path);
    (*env)->ReleaseStringUTFChars(env, model_path_string, model_path);
    if (ok) return (*env)->NewStringUTF(env, Z.backend_name);
    char result[560];
    snprintf(result, sizeof(result), "ERROR: %s", Z.last_error[0] ? Z.last_error : "QNN initialization failed");
    return (*env)->NewStringUTF(env, result);
}

JNIEXPORT jboolean JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativeProcess(
    JNIEnv *env, jobject ignored,
    jobject y_buffer, jobject u_buffer, jobject v_buffer,
    jint width, jint height,
    jint y_row_stride, jint u_row_stride, jint v_row_stride,
    jint u_pixel_stride, jint v_pixel_stride,
    jint y_offset, jint u_offset, jint v_offset,
    jint rotation, jintArray output_argb, jfloatArray metrics) {
    (void)ignored;
    if (!Z.ready || !Z.session || width <= 0 || height <= 0) return JNI_FALSE;
    if ((*env)->GetArrayLength(env, output_argb) < MODEL_PIXELS ||
        (*env)->GetArrayLength(env, metrics) < 6) return JNI_FALSE;

    const uint8_t *y_plane = (*env)->GetDirectBufferAddress(env, y_buffer);
    const uint8_t *u_plane = (*env)->GetDirectBufferAddress(env, u_buffer);
    const uint8_t *v_plane = (*env)->GetDirectBufferAddress(env, v_buffer);
    if (!y_plane || !u_plane || !v_plane) return JNI_FALSE;
    rotation = ((rotation % 360) + 360) % 360;
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) rotation = 0;

    // Trace the first frames stage-by-stage so a stalled/failed inference is
    // pinpointable from logcat (which stage prints last is where it stopped).
    static int process_count = 0;
    int frame = process_count++;
    int trace = frame < 5 || (frame % 20) == 0;
    if (trace) LOGI("nativeProcess[%d]: enter %dx%d rot=%d", frame, width, height, rotation);

    double start = milliseconds();
    preprocess_yuv(y_plane, u_plane, v_plane, width, height,
                   y_row_stride, u_row_stride, v_row_stride,
                   u_pixel_stride, v_pixel_stride,
                   y_offset, u_offset, v_offset, rotation);
    double after_preprocess = milliseconds();
    return run_preprocessed(env, output_argb, metrics, start,
                            after_preprocess, frame, trace);
}

JNIEXPORT jboolean JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativePrepareRgb(
    JNIEnv *env, jobject ignored,
    jobject y_buffer, jobject u_buffer, jobject v_buffer,
    jint width, jint height,
    jint y_row_stride, jint u_row_stride, jint v_row_stride,
    jint u_pixel_stride, jint v_pixel_stride,
    jint y_offset, jint u_offset, jint v_offset,
    jint rotation, jbyteArray output_rgb) {
    (void)ignored;
    if (width <= 0 || height <= 0 ||
        (*env)->GetArrayLength(env, output_rgb) < MODEL_PIXELS * 3) return JNI_FALSE;

    const uint8_t *y_plane = (*env)->GetDirectBufferAddress(env, y_buffer);
    const uint8_t *u_plane = (*env)->GetDirectBufferAddress(env, u_buffer);
    const uint8_t *v_plane = (*env)->GetDirectBufferAddress(env, v_buffer);
    if (!y_plane || !u_plane || !v_plane) return JNI_FALSE;
    rotation = ((rotation % 360) + 360) % 360;
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) rotation = 0;

    jbyte *rgb = (*env)->GetByteArrayElements(env, output_rgb, NULL);
    if (!rgb) return JNI_FALSE;
    UprightMap map = upright_map(width, height, rotation);
    for (int model_y = 0; model_y < MODEL_H; ++model_y) {
        for (int model_x = 0; model_x < MODEL_W; ++model_x) {
            float r, g, b;
            sample_upright_rgb(y_plane, u_plane, v_plane, &map,
                               y_row_stride, u_row_stride, v_row_stride,
                               u_pixel_stride, v_pixel_stride,
                               y_offset, u_offset, v_offset,
                               model_x, model_y, &r, &g, &b);
            int index = (model_y * MODEL_W + model_x) * 3;
            rgb[index] = (jbyte)(uint8_t)(r + 0.5f);
            rgb[index + 1] = (jbyte)(uint8_t)(g + 0.5f);
            rgb[index + 2] = (jbyte)(uint8_t)(b + 0.5f);
        }
    }
    (*env)->ReleaseByteArrayElements(env, output_rgb, rgb, 0);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativeProcessRgb(
    JNIEnv *env, jobject ignored, jbyteArray input_rgb,
    jintArray output_argb, jfloatArray metrics) {
    (void)ignored;
    if (!Z.ready || !Z.session ||
        (*env)->GetArrayLength(env, input_rgb) < MODEL_PIXELS * 3 ||
        (*env)->GetArrayLength(env, output_argb) < MODEL_PIXELS ||
        (*env)->GetArrayLength(env, metrics) < 6) return JNI_FALSE;

    static int process_count = 0;
    int frame = process_count++;
    int trace = frame < 5 || (frame % 20) == 0;
    if (trace) LOGI("nativeProcess[%d]: enter remote RGB888", frame);
    double start = milliseconds();
    jbyte *rgb = (*env)->GetByteArrayElements(env, input_rgb, NULL);
    if (!rgb) return JNI_FALSE;
    preprocess_rgb888((const uint8_t *)rgb);
    (*env)->ReleaseByteArrayElements(env, input_rgb, rgb, JNI_ABORT);
    double after_preprocess = milliseconds();
    return run_preprocessed(env, output_argb, metrics, start,
                            after_preprocess, frame, trace);
}

// Cheap YUV -> upright, center-cropped RGB with no NPU. Runs on the camera
// thread for every frame so the RGB preview stays live and correctly oriented
// regardless of whether the model is ready or the NPU worker is busy. It shows
// the exact square the model consumes, so preview and depth are framed the same.
JNIEXPORT jboolean JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativePreview(
    JNIEnv *env, jobject ignored,
    jobject y_buffer, jobject u_buffer, jobject v_buffer,
    jint width, jint height,
    jint y_row_stride, jint u_row_stride, jint v_row_stride,
    jint u_pixel_stride, jint v_pixel_stride,
    jint y_offset, jint u_offset, jint v_offset,
    jint rotation, jintArray output_argb) {
    (void)ignored;
    if (width <= 0 || height <= 0) return JNI_FALSE;
    if ((*env)->GetArrayLength(env, output_argb) < MODEL_PIXELS) return JNI_FALSE;

    const uint8_t *y_plane = (*env)->GetDirectBufferAddress(env, y_buffer);
    const uint8_t *u_plane = (*env)->GetDirectBufferAddress(env, u_buffer);
    const uint8_t *v_plane = (*env)->GetDirectBufferAddress(env, v_buffer);
    if (!y_plane || !u_plane || !v_plane) return JNI_FALSE;
    rotation = ((rotation % 360) + 360) % 360;
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) rotation = 0;

    UprightMap map = upright_map(width, height, rotation);
    for (int model_y = 0; model_y < MODEL_H; ++model_y) {
        for (int model_x = 0; model_x < MODEL_W; ++model_x) {
            float r, g, b;
            sample_upright_rgb(y_plane, u_plane, v_plane, &map,
                               y_row_stride, u_row_stride, v_row_stride,
                               u_pixel_stride, v_pixel_stride,
                               y_offset, u_offset, v_offset,
                               model_x, model_y, &r, &g, &b);
            uint32_t argb = 0xFF000000u |
                            ((uint32_t)(int)(r + 0.5f) << 16) |
                            ((uint32_t)(int)(g + 0.5f) << 8) |
                            (uint32_t)(int)(b + 0.5f);
            preview_pixels[model_y * MODEL_W + model_x] = (jint)argb;
        }
    }
    (*env)->SetIntArrayRegion(env, output_argb, 0, MODEL_PIXELS, preview_pixels);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_org_zipdepth_npudemo_ZipDepthNative_nativeShutdown(
    JNIEnv *env, jobject ignored) {
    (void)env;
    (void)ignored;
    Z.ready = 0;
    if (Z.api) {
        if (Z.run_options) Z.api->ReleaseRunOptions(Z.run_options);
        if (Z.session) Z.api->ReleaseSession(Z.session);
        if (Z.memory_info) Z.api->ReleaseMemoryInfo(Z.memory_info);
        if (Z.env) Z.api->ReleaseEnv(Z.env);
    }
    Z.run_options = NULL;
    Z.session = NULL;
    Z.memory_info = NULL;
    Z.env = NULL;
    Z.api = NULL;
    if (Z.ort_library) dlclose(Z.ort_library);
    Z.ort_library = NULL;
}
