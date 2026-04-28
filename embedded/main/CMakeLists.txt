idf_component_register(
    SRCS "main.cc" "model_data.cc"
    PRIV_REQUIRES esp_driver_i2s esp_driver_gpio esp-tflite-micro spi_flash esp_timer esp-dsp
    INCLUDE_DIRS "."
)