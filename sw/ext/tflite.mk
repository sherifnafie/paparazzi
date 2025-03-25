# TensorFlow Lite configuration

ifneq ($(PAPARAZZI_SRC),)
  TFLITE_PATH = $(PAPARAZZI_SRC)/sw/ext/tflite
  TFLITE_INCLUDES = -I$(TFLITE_PATH)
  TFLITE_LIB = $(TFLITE_PATH)/libtensorflowlite.a
endif

# Add TFLite to Paparazzi build process
$(TARGET).CFLAGS += $(TFLITE_INCLUDES)
$(TARGET).LDFLAGS += $(TFLITE_LIB) -lstdc++
