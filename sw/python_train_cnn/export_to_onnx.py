#!/usr/bin/env python3

import os
import sys
import tensorflow as tf
import numpy as np
import argparse

# Disable GPU to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tf2onnx
except ImportError:
    print("Error: tf2onnx package not installed")
    print("Please install it with: pip install tf2onnx")
    sys.exit(1)

def intra_pred_variation(y_true, y_pred):
    """Metric to track variation within each prediction"""
    def reshape_branch():
        y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
        y_pred_norm = y_pred_reshaped / 255.0
        return tf.math.reduce_variance(y_pred_norm, axis=[1, 2, 3])
    
    def direct_branch():
        # For already shaped input
        y_pred_norm = y_pred / 255.0
        # This is a simple fallback for unexpected dimensions
        return tf.reduce_mean(tf.square(y_pred_norm))
    
    # Use tf.cond to select the right branch
    variance = tf.cond(
        tf.equal(tf.rank(y_pred), 2),
        true_fn=reshape_branch,
        false_fn=direct_branch
    )
    
    return tf.reduce_mean(variance)

def inter_pred_variation(y_true, y_pred):
    """Metric to track variation between different predictions in batch"""
    def reshape_branch():
        y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
        y_pred_norm = y_pred_reshaped / 255.0
        mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
        return tf.reduce_mean(tf.square(y_pred_norm - mean_pred))
    
    def direct_branch():
        # For already shaped input
        y_pred_norm = y_pred / 255.0
        mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
        return tf.reduce_mean(tf.square(y_pred_norm - mean_pred))
    
    # Use tf.cond to select the right branch
    return tf.cond(
        tf.equal(tf.rank(y_pred), 2),
        true_fn=reshape_branch,
        false_fn=direct_branch
    )

def gradient_loss(y_true, y_pred):
    """Gradient-based loss for sharper boundaries, scaled for 0-255 range"""
    # First check input ranks and reshape if needed
    def reshape_branch():
        y_true_reshaped = tf.reshape(y_true, [-1, 12, 32, 1])
        y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
        
        # Normalize to 0-1 range for gradient calculations
        y_true_norm = y_true_reshaped / 255.0
        y_pred_norm = y_pred_reshaped / 255.0
        
        # Horizontal gradients
        grad_true_x = y_true_norm[:, :, 1:, :] - y_true_norm[:, :, :-1, :]
        grad_pred_x = y_pred_norm[:, :, 1:, :] - y_pred_norm[:, :, :-1, :]
        
        # Vertical gradients
        grad_true_y = y_true_norm[:, 1:, :, :] - y_true_norm[:, :-1, :, :]
        grad_pred_y = y_pred_norm[:, 1:, :, :] - y_pred_norm[:, :-1, :, :]
        
        # L1 difference of gradients
        loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
        loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
        
        return loss_x + loss_y
    
    def direct_branch():
        # For 4D input that's already properly shaped
        # Normalize to 0-1 range
        y_true_norm = y_true / 255.0
        y_pred_norm = y_pred / 255.0
        
        # Return a simple MSE loss for unexpected dimensions
        # This is a fallback to avoid errors
        return tf.reduce_mean(tf.square(y_true_norm - y_pred_norm))
    
    # Use tf.cond to select the right branch based on rank
    return tf.cond(
        tf.equal(tf.rank(y_true), 2),  # If rank is 2 (flattened)
        true_fn=reshape_branch,        # Use reshape branch
        false_fn=direct_branch         # Otherwise use direct branch
    )

def custom_obstacle_weighted_loss(var_penalty_weight, var_penalty_exponent, 
                                 diversity_penalty_weight, diversity_penalty_exponent, 
                                 alpha_grad):
    """Custom loss function with support for flat tensors"""
    
    def loss_function(y_true, y_pred):
        # Handle flattened inputs (rank 2)
        def reshape_branch():
            # Reshape to [batch, 12, 32, 1]
            y_true_reshaped = tf.reshape(y_true, [-1, 12, 32, 1])
            y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
            
            # Normalize to 0-1 range
            return y_true_reshaped / 255.0, y_pred_reshaped / 255.0
        
        # Handle already shaped inputs (rank 4)
        def direct_branch():
            return y_true / 255.0, y_pred / 255.0
        
        # Use tf.cond to select the right branch based on rank
        y_true_norm, y_pred_norm = tf.cond(
            tf.equal(tf.rank(y_true), 2),  # If rank is 2 (flattened)
            true_fn=reshape_branch,        # Use reshape branch
            false_fn=direct_branch         # Otherwise normalize directly
        )
        
        # Main weighted MSE
        obstacle_weight = 4.0  
        safe_weight = 1.0
        
        weights = safe_weight + (obstacle_weight - safe_weight) * tf.square(1 - y_true_norm)
        weighted_mse = tf.reduce_mean(weights * tf.square(y_true_norm - y_pred_norm))

        # Gradient-based boundary term
        g_loss = gradient_loss(y_true, y_pred)

        # Variance penalties with tunable parameters
        intra_pred_var = tf.math.reduce_variance(y_pred_norm, axis=[1, 2, 3])
        var_penalty = tf.reduce_mean(tf.exp(var_penalty_exponent * intra_pred_var))
        
        mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
        inter_pred_var = tf.reduce_mean(tf.square(y_pred_norm - mean_pred))
        diversity_penalty = tf.exp(diversity_penalty_exponent * inter_pred_var)

        # Total loss with tunable weights
        total_loss = weighted_mse \
                    + alpha_grad * g_loss \
                    + var_penalty_weight * var_penalty \
                    + diversity_penalty_weight * diversity_penalty

        return total_loss
    
    return loss_function

def scaled_sigmoid(x):
    """Custom activation that applies sigmoid and scales to 0-255 range"""
    return 255.0 * tf.nn.sigmoid(x)

def convert_model(input_model_path, output_model_path):
    """Convert TensorFlow model to ONNX format"""
    print(f"Loading TensorFlow model from {input_model_path}...")
    
    # Create loss function with default parameters from training
    default_loss_fn = custom_obstacle_weighted_loss(
        var_penalty_weight=0.12, 
        var_penalty_exponent=-15.0,
        diversity_penalty_weight=0.18, 
        diversity_penalty_exponent=-20.0,
        alpha_grad=0.2
    )
    
    # Load the model with all necessary custom objects
    custom_objects = {
        'custom_obstacle_weighted_loss': custom_obstacle_weighted_loss,
        'gradient_loss': gradient_loss,
        'intra_pred_variation': intra_pred_variation,
        'inter_pred_variation': inter_pred_variation,
        'loss_function': default_loss_fn,
        'scaled_sigmoid': scaled_sigmoid  # Add the custom activation function to custom_objects
    }
    
    # Try loading with compile=False first to avoid loss function issues
    try:
        model = tf.keras.models.load_model(input_model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with compile=False")
    except Exception as e:
        print(f"Warning: Failed to load model with compile=False: {e}")
        print("Trying to load with compile=True...")
        model = tf.keras.models.load_model(input_model_path, custom_objects=custom_objects)
        
    model.summary()
    
    # Create sample input with the right shape and value range (0-255)
    # Our model expects YUV images with values in 0-255 range
    sample_input = np.random.uniform(0, 255, (1, 240, 520, 3)).astype(np.float32)
    
    # Specify input and output names - the output will be a flat array of 384 elements
    input_signature = [tf.TensorSpec(sample_input.shape, tf.float32, name='input')]
    output_path = output_model_path
    
    # Convert the model to ONNX
    print(f"Converting model to ONNX format...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13, output_path=output_path)
    
    print(f"Model successfully converted and saved to {output_model_path}")
    print(f"Output shape: flat array of 384 elements (12x32 grid)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to ONNX format")
    parser.add_argument('-i', '--input', default='/home/lievijn/training_share/hyperopt_results/20250328_003714/models/trial_29_model.h5', help='Path to input TensorFlow model')
    parser.add_argument('-o', '--output', default='cnn_model.onnx', help='Path to output ONNX model')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU (disabled by default)')
    
    args = parser.parse_args()
    
    # Re-enable GPU if requested
    if args.gpu:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print("GPU enabled for conversion")
    else:
        print("Using CPU for conversion (GPU disabled)")
    
    # Ensure input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input model file {args.input} not found")
        sys.exit(1)
    
    # Convert model
    success = convert_model(args.input, args.output)
    
    if success:
        print("Conversion complete!")
        print(f"You can now use the ONNX model with: python evaluate_checkpoint.py --onnx")
    else:
        print("Conversion failed!")
        sys.exit(1)