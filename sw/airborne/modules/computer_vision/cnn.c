cnn_module.c

#include "cnn.h"
#include "tensorflow/lite/c/c_api.h"
#include <stdio.h>

// Model data (converted .tflite model stored as C array)
extern const unsigned char cnn_model[];
extern const unsigned int cnn_model_len;

static TfLiteModel *model = NULL;
static TfLiteInterpreterOptions *options = NULL;
static TfLiteInterpreter *interpreter = NULL;

void cnn_module_init(void) {
    model = TfLiteModelCreate(cnn_model, cnn_model_len);
    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    interpreter = TfLiteInterpreterCreate(model, options);
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        printf("Failed to allocate tensors!\n");
    } else {
        printf("CNN Module initialized successfully.\n");
    }
}

void cnn_module_run(void) {
    // Example: accessing input tensor
    float* input = TfLiteInterpreterGetInputTensor(interpreter, 0)->data.f;
    
    // Fill input tensor with your data
    // for example:
    // input[0] = ...

    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        printf("Inference failed!\n");
        return;
    }

    // Get output tensor
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    const float* output = output_tensor->data.f;

    // Use your output data here
    printf("Inference output: %f\n", output[0]);
}

// Cleanup if needed
void cnn_module_cleanup(void) {
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}
