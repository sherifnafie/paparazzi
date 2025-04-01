# Special evaluation script that processes a single JPEG image for inference

# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import argparse
from tensorflow.keras import layers, models, regularizers

# HARDCODED IMAGE PATH
# Edit this path to point to your JPEG image
IMAGE_PATH = "yuv-test.jpeg"

# Output directory
OUTPUT_FOLDER = "single_image_test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model with single JPEG image')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX model instead of TensorFlow model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model file (TF or ONNX)')
    return parser.parse_args()

args = parse_args()

# Set default model paths if not specified
if args.model_path is None:
    if args.onnx:
        args.model_path = "cnn_model.onnx"
    else:
        args.model_path = "/home/pc-sherif-2/Documents/MAV/paparazzi/playground/hyperopt_results/20250324_180533/models/trial_4_model.h5"

# Check for ONNX Runtime only if --onnx flag is used
if args.onnx:
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
    except ImportError:
        print("Error: ONNX Runtime not installed but --onnx flag was used.")
        print("Please install ONNX Runtime using:")
        print("  pip install onnxruntime")
        print("\nAlternatively, run the installation script:")
        print("  bash /home/pc-sherif-2/Documents/MAV/install_onnxruntime.sh")
        sys.exit(1)

# Set paths and directories
TF_MODEL_PATH = args.model_path
ONNX_MODEL_PATH = args.model_path

# Check if the image file exists
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file '{IMAGE_PATH}' not found!")
    print("Please ensure the image file exists at the specified path.")
    sys.exit(1)

# Check if the model file exists before proceeding
if args.onnx and not os.path.exists(ONNX_MODEL_PATH):
    print(f"Error: ONNX model file '{ONNX_MODEL_PATH}' not found!")
    print("Please ensure the model file exists in the current directory.")
    sys.exit(1)
elif not args.onnx and not os.path.exists(TF_MODEL_PATH):
    print(f"Error: TensorFlow model file '{TF_MODEL_PATH}' not found!")
    print("Please ensure the model file exists in the correct path.")
    sys.exit(1)

# Load and preprocess the image
def load_image(image_path):
    print(f"Loading image from {image_path}...")
    # Read image
    rgb_img = cv2.imread(image_path)
    if rgb_img is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Resize to model's expected input size (240x520)
    rgb_img = cv2.resize(rgb_img, (520, 240))
    
    # Convert to float32 without normalizing
    rgb_img = rgb_img.astype(np.float32)
    
    # Convert to YUV - normalize temporarily for conversion, then scale back
    yuv_img = tf.image.rgb_to_yuv(rgb_img / 255.0).numpy() * 255.0
    
    return rgb_img, yuv_img

# Define metric functions needed for model loading
def intra_pred_variation(y_true, y_pred):
    """Metric to track variation within each prediction"""
    return tf.reduce_mean(tf.math.reduce_variance(y_pred, axis=[1, 2, 3]))

def inter_pred_variation(y_true, y_pred):
    """Metric to track variation between different predictions in batch"""
    mean_pred = tf.reduce_mean(y_pred, axis=0, keepdims=True)
    return tf.reduce_mean(tf.square(y_pred - mean_pred))

def gradient_loss(y_true, y_pred):
    """Gradient-based loss for sharper boundaries"""
    # Horizontal gradients
    grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

    # Vertical gradients
    grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]

    # L1 difference of gradients
    loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
    loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))

    return loss_x + loss_y

def custom_obstacle_weighted_loss(var_penalty_weight, var_penalty_exponent, 
                                 diversity_penalty_weight, diversity_penalty_exponent, 
                                 alpha_grad, obstacle_weight=4.0):
    """Create a custom loss function with tunable hyperparameters"""
    
    def loss_function(y_true, y_pred):
        # Main weighted MSE
        safe_weight = 1.0
        
        weights = safe_weight + (obstacle_weight - safe_weight) * tf.square(1 - y_true)
        weighted_mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))

        # Gradient-based boundary term
        g_loss = gradient_loss(y_true, y_pred)

        # Variance penalties with tunable parameters
        intra_pred_var = tf.math.reduce_variance(y_pred, axis=[1, 2, 3])
        var_penalty = tf.reduce_mean(tf.exp(var_penalty_exponent * intra_pred_var))
        
        mean_pred = tf.reduce_mean(y_pred, axis=0, keepdims=True)
        inter_pred_var = tf.reduce_mean(tf.square(y_pred - mean_pred))
        diversity_penalty = tf.exp(diversity_penalty_exponent * inter_pred_var)

        # Total loss with tunable weights
        total_loss = weighted_mse \
                    + alpha_grad * g_loss \
                    + var_penalty_weight * var_penalty \
                    + diversity_penalty_weight * diversity_penalty

        return total_loss
    
    return loss_function

# Add model architecture definitions that might be needed for loading custom models
def build_model_var1(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3, obstacle_weight=4.0):
    loss_fn = custom_obstacle_weighted_loss(
        var_penalty_weight=0.12,
        var_penalty_exponent=-15.0,
        diversity_penalty_weight=0.18,
        diversity_penalty_exponent=-20.0,
        alpha_grad=0.2,
        obstacle_weight=obstacle_weight
    )
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # Encoder (modified first layer)
    x = layers.Conv2D(18, (5, 5), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(24, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(48, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(96, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Decoder
    x = layers.Conv2DTranspose(48, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(24, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(12, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    x = layers.Resizing(12, 32)(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var2(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3, obstacle_weight=4.0):
    loss_fn = custom_obstacle_weighted_loss(
        var_penalty_weight=0.12,
        var_penalty_exponent=-15.0,
        diversity_penalty_weight=0.18,
        diversity_penalty_exponent=-20.0,
        alpha_grad=0.2,
        obstacle_weight=obstacle_weight
    )
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # Encoder
    x = layers.Conv2D(12, (7, 7), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(36, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(24, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(96, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Decoder
    x = layers.Conv2DTranspose(48, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(24, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(12, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    x = layers.Resizing(12, 32)(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_efficient_resnet(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3, obstacle_weight=4.0):
    """
    Efficient Residual CNN:
      - Encoder uses a standard conv then a grouped conv,
        followed by a residual block.
      - Decoder uses a 1x1 conv followed by resizing.
    """
    loss_fn = custom_obstacle_weighted_loss(
        var_penalty_weight=0.12,
        var_penalty_exponent=-15.0,
        diversity_penalty_weight=0.18,
        diversity_penalty_exponent=-20.0,
        alpha_grad=0.2,
        obstacle_weight=obstacle_weight
    )
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # Encoder: Layer 1: Conv2D with 12 filters, 5x5, stride 4.
    x = layers.Conv2D(12, (5, 5), strides=(4, 4),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output shape: 60x130x12

    # Layer 2: Grouped Conv2D with 24 filters, 3x3, stride 2, groups=2.
    x = layers.Conv2D(24, (3, 3), strides=(2, 2),
                      padding='same', activation='relu',
                      groups=2, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output shape: 30x65x24

    # Residual block:
    x_skip = x
    x = layers.Conv2D(24, (3, 3), padding='same',
                      activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(24, (3, 3), padding='same',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, x_skip])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Decoder: 1x1 conv then resize to 12x32.
    x = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    x = layers.Resizing(12, 32)(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_depthwise_separable(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3, obstacle_weight=4.0):
    """
    Depthwise-Separable CNN:
      - Encoder uses standard conv then a sequence of separable conv layers.
      - A small residual branch is applied at the low-resolution feature map.
      - Decoder: 1x1 conv and resizing.
    """
    loss_fn = custom_obstacle_weighted_loss(
        var_penalty_weight=0.12,
        var_penalty_exponent=-15.0,
        diversity_penalty_weight=0.18,
        diversity_penalty_exponent=-20.0,
        alpha_grad=0.2,
        obstacle_weight=obstacle_weight
    )
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # Layer 1: Standard Conv2D, 16 filters, 3x3, stride 2.
    x = layers.Conv2D(16, (3, 3), strides=(2, 2),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output: 120x260x16

    # Layer 2: SeparableConv2D, 32 filters, 3x3, stride 2.
    x = layers.SeparableConv2D(32, (3, 3), strides=(2, 2),
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output: 60x130x32

    # Layer 3: SeparableConv2D, 48 filters, 3x3, stride 2.
    x = layers.SeparableConv2D(48, (3, 3), strides=(2, 2),
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output: 30x65x48

    # Layer 4: SeparableConv2D, 64 filters, 3x3, stride 2.
    x = layers.SeparableConv2D(64, (3, 3), strides=(2, 2),
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # Output: ~15x33x64

    # Residual block on low-res feature map:
    x_skip = x
    x = layers.SeparableConv2D(64, (3, 3), padding='same',
                               activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.SeparableConv2D(64, (3, 3), padding='same',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, x_skip])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Decoder: 1x1 conv then resizing.
    x = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    x = layers.Resizing(12, 32)(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', intra_pred_variation, inter_pred_variation]
    )
    return model

# Load model based on the selected type
def load_model():
    if args.onnx:
        print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
        try:
            # Create ONNX runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(ONNX_MODEL_PATH, session_options)
            
            # Print model input and output details
            print("Model inputs:")
            for i, input_info in enumerate(session.get_inputs()):
                print(f"  Input #{i}: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
            
            print("Model outputs:")
            for i, output_info in enumerate(session.get_outputs()):
                print(f"  Output #{i}: name='{output_info.name}', shape={output_info.shape}, type={output_info.type}")
                
            return session
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")
            sys.exit(1)
    else:
        print(f"Loading TensorFlow model from {TF_MODEL_PATH}...")
        try:
            # Create custom objects dictionary with all necessary functions
            custom_objects = {
                'custom_obstacle_weighted_loss': custom_obstacle_weighted_loss,
                'gradient_loss': gradient_loss,
                'intra_pred_variation': intra_pred_variation,
                'inter_pred_variation': inter_pred_variation,
                'build_model_var1': build_model_var1,
                'build_model_var2': build_model_var2,
                'build_model_efficient_resnet': build_model_efficient_resnet,
                'build_model_depthwise_separable': build_model_depthwise_separable
            }
            
            # Default loss function used during training
            default_loss_fn = custom_obstacle_weighted_loss(
                var_penalty_weight=0.12, 
                var_penalty_exponent=-15.0,
                diversity_penalty_weight=0.18, 
                diversity_penalty_exponent=-20.0,
                alpha_grad=0.2,
                obstacle_weight=4.0
            )
            
            custom_objects['loss_function'] = default_loss_fn
            
            # Try loading with compile=False first
            try:
                model = tf.keras.models.load_model(TF_MODEL_PATH, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully with compile=False")
            except Exception as e:
                print(f"Warning: Failed to load model with compile=False: {e}")
                print("Trying to load with compile=True...")
                model = tf.keras.models.load_model(TF_MODEL_PATH, custom_objects=custom_objects)
                
            model.summary()
            return model
        except Exception as e:
            print(f"Error loading TensorFlow model: {str(e)}")
            sys.exit(1)

# Load the appropriate model
model = load_model()

# Load the JPEG image and convert to YUV
rgb_image, yuv_image = load_image(IMAGE_PATH)

# Run inference based on model type
def run_inference(model, sample):
    if args.onnx:
        print("Running ONNX inference...")
        try:
            # Get input name from the model
            input_name = model.get_inputs()[0].name
            
            # Add batch dimension (ONNX models typically expect NCHW or NHWC format)
            input_data = np.expand_dims(sample, axis=0).astype(np.float32)
            
            # Run inference
            pred = model.run(None, {input_name: input_data})[0]
            
            return pred[0]  # Remove batch dimension
        except Exception as e:
            print(f"Error during ONNX inference: {str(e)}")
            sys.exit(1)
    else:
        print("Running TensorFlow inference...")
        # Add batch dimension for model input
        input_data = np.expand_dims(sample, axis=0)
        pred = model.predict(input_data)
        return pred[0]  # Remove batch dimension

# Run inference with our image
print(f"Running inference with image from {IMAGE_PATH}...")
prediction = run_inference(model, yuv_image)

# Save the safety grid as a text file
# Using plain text format for better readability
grid_path = os.path.join(OUTPUT_FOLDER, "safety_grid.txt")
print(f"Saving safety grid to {grid_path}")

with open(grid_path, 'w') as f:
    # Write header with information
    f.write(f"# Safety Grid Results\n")
    f.write(f"# Model: {args.model_path}\n")
    f.write(f"# Image: {IMAGE_PATH}\n")
    f.write(f"# Grid shape: {prediction.shape}\n\n")
    
    # Write the grid values with row/column indices for clarity
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            f.write(f"{prediction[i, j][0]:.6f} ")
        f.write("\n")

print(f"Inference completed successfully!")
print(f"Results saved to {grid_path}")
