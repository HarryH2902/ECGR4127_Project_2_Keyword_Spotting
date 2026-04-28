#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"

#include "dsps_fft2r.h"

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "keyword_stream";

// Audio settings
#define SAMPLE_RATE          16000
#define CLIP_SAMPLES         24000      // 1.5 seconds at 16 kHz
#define READ_SAMPLES         1024
#define STREAM_STEP_SAMPLES  8000       // 0.5 seconds at 16 kHz

// STFT settings from training
#define FRAME_LENGTH         480
#define FRAME_STEP           320
#define FFT_LENGTH           512
#define NUM_FRAMES           74
#define NUM_BINS             257

// Best live mic gain so far
#define MIC_GAIN             10.0f

// Detection settings
#define FLYING_THRESHOLD     0.30f
#define CLASS_MARGIN         0.10f
#define REQUIRED_HITS        2
#define DETECTION_COOLDOWN   3

// ESP32-S3-EYE-MB v2.2 mic pins
#define I2S_MIC_BCLK         GPIO_NUM_41
#define I2S_MIC_WS           GPIO_NUM_42
#define I2S_MIC_DIN          GPIO_NUM_2

constexpr int kTensorArenaSize = 512 * 1024;
static const float PI_F = 3.14159265358979323846f;

static i2s_chan_handle_t rx_chan;

// Static buffers avoid stack overflow
static int32_t i2s_read_buffer[READ_SAMPLES];

// Rolling raw audio buffer
static int16_t audio_clip[CLIP_SAMPLES];

// Gain-adjusted copy used for model preprocessing
static int16_t audio_proc[CLIP_SAMPLES];

static float hann_window[FRAME_LENGTH];

// Complex FFT buffer: real, imag, real, imag...
static float fft_buffer[FFT_LENGTH * 2];

static const char *labels[] = {
    "background",
    "unknown",
    "flying",
    "happy"
};

static int16_t clamp_to_int16(int32_t value)
{
    if (value > 32767) return 32767;
    if (value < -32768) return -32768;
    return (int16_t)value;
}

static void make_hann_window(void)
{
    for (int n = 0; n < FRAME_LENGTH; n++) {
        hann_window[n] = 0.5f - 0.5f * cosf((2.0f * PI_F * n) / (FRAME_LENGTH - 1));
    }
}

static void i2s_mic_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_slot_config_t slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_MONO
    );

    slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = slot_cfg,
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_BCLK,
            .ws = I2S_MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_MIC_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));

    ESP_LOGI(TAG, "I2S microphone initialized.");
    ESP_LOGI(TAG, "BCLK=%d WS=%d DIN=%d sample_rate=%d",
             I2S_MIC_BCLK, I2S_MIC_WS, I2S_MIC_DIN, SAMPLE_RATE);
}

static bool capture_samples_into_buffer(int16_t *dest, int sample_count_needed)
{
    int captured = 0;

    while (captured < sample_count_needed) {
        size_t bytes_read = 0;

        esp_err_t err = i2s_channel_read(
            rx_chan,
            i2s_read_buffer,
            sizeof(i2s_read_buffer),
            &bytes_read,
            pdMS_TO_TICKS(1000)
        );

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(err));
            return false;
        }

        int samples_read = bytes_read / sizeof(int32_t);

        for (int i = 0; i < samples_read && captured < sample_count_needed; i++) {
            int32_t s = i2s_read_buffer[i] >> 14;
            dest[captured] = clamp_to_int16(s);
            captured++;
        }
    }

    return true;
}

static bool fill_initial_audio_buffer(void)
{
    ESP_LOGI(TAG, "Filling initial 1.5-second rolling buffer...");
    bool ok = capture_samples_into_buffer(audio_clip, CLIP_SAMPLES);

    if (ok) {
        ESP_LOGI(TAG, "Initial rolling buffer ready.");
    }

    return ok;
}

static bool update_rolling_audio_buffer(void)
{
    memmove(
        audio_clip,
        audio_clip + STREAM_STEP_SAMPLES,
        (CLIP_SAMPLES - STREAM_STEP_SAMPLES) * sizeof(int16_t)
    );

    int16_t *tail = audio_clip + (CLIP_SAMPLES - STREAM_STEP_SAMPLES);

    return capture_samples_into_buffer(tail, STREAM_STEP_SAMPLES);
}

static void make_gain_copy(float gain)
{
    ESP_LOGI(TAG, "Applying mic gain copy: %.2fx", gain);

    for (int i = 0; i < CLIP_SAMPLES; i++) {
        int32_t amplified = (int32_t)((float)audio_clip[i] * gain);
        audio_proc[i] = clamp_to_int16(amplified);
    }
}

static void print_audio_stats(const char *name, const int16_t *buffer)
{
    int16_t min_val = INT16_MAX;
    int16_t max_val = INT16_MIN;
    int64_t sum = 0;
    double sum_sq = 0.0;

    for (int i = 0; i < CLIP_SAMPLES; i++) {
        int16_t s = buffer[i];

        if (s < min_val) min_val = s;
        if (s > max_val) max_val = s;

        sum += s;
        sum_sq += (double)s * (double)s;
    }

    double mean = (double)sum / (double)CLIP_SAMPLES;
    double rms = sqrt(sum_sq / (double)CLIP_SAMPLES);

    ESP_LOGI(TAG, "%s stats: min=%d max=%d mean=%.2f rms=%.2f",
             name, min_val, max_val, mean, rms);
}

static void audio_to_model_input(TfLiteTensor *input, const int16_t *buffer)
{
    ESP_LOGI(TAG, "Converting audio to spectrogram input using FFT...");

    int16_t max_abs = 1;

    for (int i = 0; i < CLIP_SAMPLES; i++) {
        int16_t value = buffer[i];
        int16_t abs_val = value >= 0 ? value : -value;

        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    ESP_LOGI(TAG, "Audio normalization max_abs=%d", max_abs);

    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    int out_index = 0;

    int64_t spec_start = esp_timer_get_time();

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int frame_start = frame * FRAME_STEP;

        // Clear FFT buffer
        for (int i = 0; i < FFT_LENGTH * 2; i++) {
            fft_buffer[i] = 0.0f;
        }

        // Fill real part with windowed audio
        for (int n = 0; n < FRAME_LENGTH; n++) {
            int sample_index = frame_start + n;

            float sample = 0.0f;

            if (sample_index < CLIP_SAMPLES) {
                sample = (float)buffer[sample_index] / (float)max_abs;
            }

            sample *= hann_window[n];

            fft_buffer[2 * n] = sample;
            fft_buffer[2 * n + 1] = 0.0f;
        }

        // Run FFT
        dsps_fft2r_fc32(fft_buffer, FFT_LENGTH);
        dsps_bit_rev_fc32(fft_buffer, FFT_LENGTH);

        // First 257 bins
        for (int bin = 0; bin < NUM_BINS; bin++) {
            float real = fft_buffer[2 * bin];
            float imag = fft_buffer[2 * bin + 1];

            float magnitude = sqrtf(real * real + imag * imag);
            float log_mag = logf(magnitude + 1e-6f);

            int q = (int)roundf(log_mag / input_scale + input_zero_point);

            if (q < -128) q = -128;
            if (q > 127) q = 127;

            input->data.int8[out_index] = (int8_t)q;
            out_index++;
        }

        if ((frame % 8) == 0) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }

    int64_t spec_end = esp_timer_get_time();

    ESP_LOGI(TAG, "FFT spectrogram conversion complete. Filled %d values.", out_index);
    ESP_LOGI(TAG, "Spectrogram time: %.2f ms", (double)(spec_end - spec_start) / 1000.0);
}

static int get_top_prediction(TfLiteTensor *output)
{
    int best_index = 0;
    int8_t best_raw = output->data.int8[0];

    for (int i = 1; i < 4; i++) {
        if (output->data.int8[i] > best_raw) {
            best_raw = output->data.int8[i];
            best_index = i;
        }
    }

    return best_index;
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting FFT streaming-style keyword spotting test...");

    make_hann_window();

    esp_err_t fft_status = dsps_fft2r_init_fc32(NULL, FFT_LENGTH);
    if (fft_status != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(fft_status));
        return;
    }

    ESP_LOGI(TAG, "FFT initialized with length %d.", FFT_LENGTH);

    i2s_mic_init();

    uint8_t *tensor_arena = (uint8_t *)heap_caps_malloc(
        kTensorArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );

    if (tensor_arena == nullptr) {
        ESP_LOGW(TAG, "PSRAM allocation failed. Trying normal heap.");
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT);
    }

    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena.");
        return;
    }

    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes", kTensorArenaSize);

    const tflite::Model *model = tflite::GetModel(keyword_model_quantized_tflite);

    if (model->version() != 3) {
        ESP_LOGE(TAG, "Model schema version %d is not supported", model->version());
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed.");
        return;
    }

    ESP_LOGI(TAG, "AllocateTensors() succeeded.");
    ESP_LOGI(TAG, "Arena used: %u bytes", (unsigned int)interpreter.arena_used_bytes());

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    ESP_LOGI(TAG, "Input shape: [%d, %d, %d, %d]",
             input->dims->data[0],
             input->dims->data[1],
             input->dims->data[2],
             input->dims->data[3]);

    ESP_LOGI(TAG, "Output shape: [%d, %d]",
             output->dims->data[0],
             output->dims->data[1]);

    if (!fill_initial_audio_buffer()) {
        ESP_LOGE(TAG, "Initial audio buffer fill failed.");
        return;
    }

    int flying_hits = 0;
    int cooldown = 0;

    ESP_LOGI(TAG, "FFT streaming-style detection started.");
    ESP_LOGI(TAG, "Say 'flying'. Inference uses the latest rolling 1.5 seconds.");

    while (true) {
        ESP_LOGI(TAG, "Listening... updating rolling window.");

        if (!update_rolling_audio_buffer()) {
            ESP_LOGE(TAG, "Rolling audio update failed.");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        print_audio_stats("Raw rolling audio", audio_clip);

        make_gain_copy(MIC_GAIN);

        print_audio_stats("Gain-adjusted audio", audio_proc);

        audio_to_model_input(input, audio_proc);

        ESP_LOGI(TAG, "Running inference...");

        int64_t start_time = esp_timer_get_time();

        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke() failed.");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        int64_t end_time = esp_timer_get_time();

        ESP_LOGI(TAG, "Invoke() succeeded. Inference time: %.2f ms",
                 (double)(end_time - start_time) / 1000.0);

        float scores[4];

        for (int i = 0; i < 4; i++) {
            int8_t raw = output->data.int8[i];
            scores[i] = output->params.scale *
                        ((float)raw - (float)output->params.zero_point);

            ESP_LOGI(TAG, "Output[%d] %-10s raw=%d score=%.4f",
                     i, labels[i], raw, scores[i]);
        }

        int pred = get_top_prediction(output);

        float background_score = scores[0];
        float unknown_score = scores[1];
        float flying_score = scores[2];
        float happy_score = scores[3];

        ESP_LOGI(TAG, "Top prediction: %s", labels[pred]);

        bool possible_flying =
            flying_score > FLYING_THRESHOLD &&
            flying_score > unknown_score + CLASS_MARGIN &&
            flying_score > happy_score + CLASS_MARGIN;

        if (cooldown > 0) {
            cooldown--;
            ESP_LOGI(TAG, "Detection cooldown active: %d", cooldown);
        } else {
            if (possible_flying) {
                flying_hits++;

                ESP_LOGW(TAG,
                         "Flying hit %d/%d: flying_score=%.4f background_score=%.4f",
                         flying_hits,
                         REQUIRED_HITS,
                         flying_score,
                         background_score);
            } else {
                flying_hits = 0;
            }

            if (flying_hits >= REQUIRED_HITS) {
                ESP_LOGW(TAG, ">>> CONFIRMED FLYING DETECTED! <<<");
                flying_hits = 0;
                cooldown = DETECTION_COOLDOWN;
            }
        }

        ESP_LOGI(TAG, "Next rolling window soon...\n");

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}