# Training script with hyperparameter optimization for obstacle avoidance CNN

###############################
# Imports
###############################
import os
import cv2
import json
import numpy as np
import pandas as pd
import random
import signal
import sys
import pickle
import time
import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import glob

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.backend import int_shape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Optimization imports
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

###############################
# Constants and Paths
###############################
# Data paths
IMAGE_FOLDER_PROVIDEDIMAGES = "provided_images"
IMAGE_FOLDER_VIDEOCAPSIMULATIONROUND1 = "videocap_simulation_round1"
IMAGE_FOLDER_TESTROUND2 = "videocap_testround2"
IMAGE_FOLDER_VIDEOCAPSIMULATIONROUND2 = "videocap_simulation_round2"

LABELS_JSON_PROVIDEDIMAGES = "labelled_provided_images/results.json"
LABELS_JSON_VIDEOCAPSIMULATIONROUND1 = "labelled_videocap_simulation_round1/results.json"
LABELS_JSON_TESTROUND2 = "labelled_videocap_testround2/results.json"
LABELS_JSON_VIDEOCAPSIMULATIONROUND2 = "labelled_videocap_simulation_round2/results.json"

OUTPUT_FOLDER = "PLACEHOLDER"
VIDEO_OUTPUT = "PLACEHOLDER"
TEXT_OUTPUT = "PLACEHOLDER"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Results and logging paths will be set in main() based on --warm flag

###############################
# Data Loading Functions
###############################
def load_data(image_folder, labels, img_size=(520, 240)):
    """Load images and their corresponding label grids"""
    images, label_grids = [], []
    for filename, data in labels.items():
        img_path = os.path.join(image_folder, f"{filename}.jpg")
        if not os.path.exists(img_path):
            continue

        # Read the image (BGR format by default in OpenCV)
        img = cv2.imread(img_path)
        
        # Convert BGR to YUV format to match evaluation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        images.append(img)

        # Get the label grid directly - already in 0-255 range in the JSON
        label_grid = np.array(data["scores"])
        label_grids.append(label_grid)

    return np.array(images), np.array(label_grids)

###############################
# Metrics and Loss Functions
###############################
def gradient_loss(y_true, y_pred):
    """Gradient-based loss for sharper boundaries, scaled for 0-255 range"""
    y_true_reshaped = tf.reshape(y_true, [-1, 12, 32, 1])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    
    y_true_norm = y_true_reshaped / 255.0
    y_pred_norm = y_pred_reshaped / 255.0
    
    # Horizontal gradients
    grad_true_x = y_true_norm[:, :, 1:, :] - y_true_norm[:, :, :-1, :]
    grad_pred_x = y_pred_norm[:, :, 1:, :] - y_pred_norm[:, :, :-1, :]
    
    # Vertical gradients
    grad_true_y = y_true_norm[:, 1:, :, :] - y_true_norm[:, :-1, :, :]
    grad_pred_y = y_pred_norm[:, 1:, :, :] - y_pred_norm[:, :-1, :, :]
    
    loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
    loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
    
    return loss_x + loss_y

def custom_obstacle_loss_v2(obstacle_weight, 
                            var_penalty_weight, 
                            diversity_penalty_weight, 
                            alpha_grad):
    """
    Revised custom loss function for obstacle avoidance.
    Uses Weighted MAE, Logarithmic Variance Penalties.
    """
    safe_weight = 1.0
    epsilon = K.epsilon()  # Small constant for numerical stability

    def loss_function(y_true, y_pred):
        # --- Reshaping and Normalization ---
        y_true_r = tf.reshape(y_true, [-1, 12, 32, 1])
        y_pred_r = tf.reshape(y_pred, [-1, 12, 32, 1])
        y_true_norm = y_true_r / 255.0
        y_pred_norm = y_pred_r / 255.0

        # --- 1. Weighted MAE Loss (Linear Weighting) ---
        weights = safe_weight + (obstacle_weight - safe_weight) * (1.0 - y_true_norm)
        weights = tf.maximum(weights, 0.0)
        weighted_mae = tf.reduce_mean(weights * tf.abs(y_true_norm - y_pred_norm))

        # --- 2. Gradient Loss (L1 difference of gradients) ---
        g_loss = gradient_loss(y_true, y_pred)

        # --- 3. Intra-Prediction Variance Penalty (Logarithmic) ---
        intra_pred_var = tf.math.reduce_variance(y_pred_norm, axis=[1, 2, 3])
        var_penalty = tf.reduce_mean(-tf.math.log(intra_pred_var + epsilon))

        # --- 4. Inter-Prediction Diversity Penalty (Logarithmic) ---
        mean_pred_norm = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
        inter_pred_var = tf.reduce_mean(tf.square(y_pred_norm - mean_pred_norm))
        diversity_penalty = -tf.math.log(inter_pred_var + epsilon)

        # --- 5. Combine Loss Terms ---
        total_loss = weighted_mae \
                     + alpha_grad * g_loss \
                     + var_penalty_weight * var_penalty \
                     + diversity_penalty_weight * diversity_penalty

        return total_loss

    return loss_function

def intra_pred_variation(y_true, y_pred):
    """Metric to track variation within each prediction"""
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    y_pred_norm = y_pred_reshaped / 255.0
    variance = tf.math.reduce_variance(y_pred_norm, axis=[1, 2, 3])
    return tf.reduce_mean(variance)

def inter_pred_variation(y_true, y_pred):
    """Metric to track variation between different predictions in batch"""
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    y_pred_norm = tf.reshape(y_pred, [-1, 12, 32, 1]) / 255.0
    mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
    squared_diffs = tf.square(y_pred_norm - mean_pred)
    return tf.reduce_mean(squared_diffs)

# New Metric: F1 Score based on a fixed threshold
def f1_score_metric(y_true, y_pred):
    """Calculate mean F1 score for obstacle detection (obstacle if value < 128)"""
    threshold = 128.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    bin_y_true = tf.cast(y_true_flat < threshold, tf.float32)
    bin_y_pred = tf.cast(y_pred_flat < threshold, tf.float32)
    true_positives = tf.reduce_sum(bin_y_true * bin_y_pred, axis=1)
    predicted_positives = tf.reduce_sum(bin_y_pred, axis=1)
    actual_positives = tf.reduce_sum(bin_y_true, axis=1)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return tf.reduce_mean(f1)

# New Metric: Structural Similarity Index (SSIM)
def ssim_metric(y_true, y_pred):
    """Calculate mean SSIM for depth map predictions"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_reshaped = tf.reshape(y_true, [tf.shape(y_true)[0], 12, 32, 1])
    y_pred_reshaped = tf.reshape(y_pred, [tf.shape(y_pred)[0], 12, 32, 1])
    y_true_norm = y_true_reshaped / 255.0
    y_pred_norm = y_pred_reshaped / 255.0
    ssim_val = tf.image.ssim(y_true_norm, y_pred_norm, max_val=1.0)
    return tf.reduce_mean(ssim_val)

###############################
# Model Architectures
###############################

def scaled_sigmoid(x):
    """Custom activation that applies sigmoid and scales to 0-255 range"""
    return 255.0 * tf.nn.sigmoid(x)

def build_model_base(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """Original encoder-decoder architecture modified to target ~25M MACs"""
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )
    
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # Encoder
    x = layers.Conv2D(10, (7, 7), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(20, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(40, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(79, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Decoder
    x = layers.Conv2DTranspose(40, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(20, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(10, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=None)(x)
    x = layers.Activation(scaled_sigmoid)(x)
    x = layers.Resizing(12, 32)(x)
    x = layers.Flatten()(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var1(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """5×5 kernel with 15 filters in first layer (scaled from 24) to target ~25M MAC"""
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )
    
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    x = layers.Conv2D(15, (5, 5), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(20, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(40, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(80, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(40, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(20, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(10, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=None)(x)
    x = layers.Activation(scaled_sigmoid)(x)
    x = layers.Resizing(12, 32)(x)
    x = layers.Flatten()(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var2(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3): 
    """Adjusted filter counts: 10→30→20→80
       Estimated total MAC ≈ 25e6 (calculated layer‐by‐layer)"""
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )
    
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    x = layers.Conv2D(10, (7, 7), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(30, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(20, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(80, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(40, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(20, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(10, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=None)(x)
    x = layers.Activation(scaled_sigmoid)(x)
    x = layers.Resizing(12, 32)(x)
    x = layers.Flatten()(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var3(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """Modified model targeting ~25 million MAC operations."""
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )
    
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    x = layers.Conv2D(10, (7, 7), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x_in = x
    r = layers.Conv2D(20, (3, 3), strides=2,
                      padding='same', kernel_regularizer=reg)(x_in)
    r = layers.BatchNormalization()(r)
    r = layers.Activation('relu')(r)
    r = layers.Conv2D(20, (3, 3), padding='same', kernel_regularizer=reg)(r)
    r = layers.BatchNormalization()(r)
    skip = layers.Conv2D(20, (1, 1), strides=2,
                         padding='same', kernel_regularizer=reg)(x_in)
    x = layers.Add()([r, skip])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(36, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(72, (3, 3), strides=2,
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(36, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(20, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(10, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=None)(x)
    x = layers.Activation(scaled_sigmoid)(x)
    x = layers.Resizing(12, 32)(x)
    x = layers.Flatten()(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var4(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """SeparableConv2D layers in encoder for efficiency"""
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )
    
    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    x = layers.Conv2D(11, (7, 7), strides=(7, 7),
                      padding='same', activation='relu',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.SeparableConv2D(21, (3, 3), strides=2,
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.SeparableConv2D(43, (3, 3), strides=2,
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.SeparableConv2D(85, (3, 3), strides=2,
                               padding='same', activation='relu',
                               depthwise_regularizer=reg,
                               pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(43, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(21, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(11, (3, 3), strides=2,
                               padding='same', activation='relu',
                               kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate/1.5)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=None)(x)
    x = layers.Activation(scaled_sigmoid)(x)
    x = layers.Resizing(12, 32)(x)
    x = layers.Flatten()(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_model_var5(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """
    U-Net like architecture with explicit skip connections. (FIXED with Cropping2D)
    Reduced filter counts to target ~25M MACs.
    Encoder Filters: 8 -> 16 -> 32 -> 64
    Decoder Filters (ConvT): 32 -> 16 -> 8
    Decoder Filters (Fusion): 32 -> 16 -> 8
    """
    # Default loss function using v2
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )

    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # --- Encoder ---
    # Stage 1
    e1 = layers.Conv2D(8, (7, 7), strides=(7, 7), padding='same', activation='relu', kernel_regularizer=reg)(inputs)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.Dropout(dropout_rate)(e1)
    # Output shape ~ (Batch, 35, 75, 8) - Keep for skip connection 1

    # Stage 2
    e2 = layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.Dropout(dropout_rate)(e2)
    # Output shape ~ (Batch, 18, 38, 16) - Keep for skip connection 2

    # Stage 3
    e3 = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.Dropout(dropout_rate)(e3)
    # Output shape ~ (Batch, 9, 19, 32) - Keep for skip connection 3

    # Bottleneck (Stage 4)
    b = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e3)
    b = layers.BatchNormalization()(b)
    b = layers.Dropout(dropout_rate)(b)
    # Output shape ~ (Batch, 5, 10, 64)

    # --- Decoder ---
    # Stage 1 (Upsample + Skip 3)
    d1 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(b)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(dropout_rate)(d1)
    # Output shape of d1 is ~ (Batch, 10, 20, 32)

    # ---> FIX: Crop d1 to match spatial dimensions of e3 <---
    # Target shape e3: (Batch, 9, 19, 32)
    # Current shape d1: (Batch, 10, 20, 32)
    # Crop 1 pixel from top, 0 from bottom -> H = 10-1-0 = 9
    # Crop 1 pixel from left, 0 from right -> W = 20-1-0 = 19
    d1 = layers.Cropping2D(cropping=((1, 0), (1, 0)), name='crop_d1_to_e3')(d1)

    # Concatenate with encoder stage 3 output
    d1 = layers.Concatenate()([d1, e3])
    # Fusion Convolution
    d1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Dropout(dropout_rate)(d1)
    # Output shape ~ (Batch, 9, 19, 32)

    # Stage 2 (Upsample + Skip 2)
    d2 = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(dropout_rate)(d2)
    # Output shape d2: ~ (Batch, 18, 38, 16) - Should match e2
    # Note: Add cropping here if necessary, but likely matches now.
    # d2 = layers.Cropping2D(cropping=((?, ?), (?, ?)), name='crop_d2_to_e2')(d2) # If needed

    # Concatenate with encoder stage 2 output
    d2 = layers.Concatenate()([d2, e2])
    # Fusion Convolution
    d2 = layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Dropout(dropout_rate)(d2)
    # Output shape ~ (Batch, 18, 38, 16)

    # Stage 3 (Upsample + Skip 1)
    d3 = layers.Conv2DTranspose(8, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(dropout_rate / 1.5)(d3)
    # Output shape d3: ~ (Batch, 36, 76, 8) - Check against e1 (35, 75, 8) -> Mismatch!

    # ---> FIX: Crop d3 to match spatial dimensions of e1 <---
    # Target shape e1: (Batch, 35, 75, 8)
    # Current shape d3: (Batch, 36, 76, 8) approx
    # Need to confirm exact output size or dynamically crop, but let's assume 36x76
    # Crop 1 pixel from top, 0 from bottom -> H = 36-1-0 = 35
    # Crop 1 pixel from left, 0 from right -> W = 76-1-0 = 75
    d3 = layers.Cropping2D(cropping=((1, 0), (1, 0)), name='crop_d3_to_e1')(d3)

    # Concatenate with encoder stage 1 output
    d3 = layers.Concatenate()([d3, e1])
    # Fusion Convolution
    d3 = layers.Conv2D(8, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(dropout_rate / 1.5)(d3)
    # Output shape ~ (Batch, 35, 75, 8)

    # --- Output Head ---
    output = layers.Conv2D(1, (3, 3), padding='same', activation=None)(d3)
    output = layers.Activation(scaled_sigmoid)(output)
    output = layers.Resizing(12, 32)(output)
    output = layers.Flatten()(output)

    model = models.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

# Helper function for Squeeze-and-Excitation block (needed for var6)
def se_block(input_tensor, ratio=8):
    """
    Creates a Squeeze-and-Excitation block.
    Args:
        input_tensor: input tensor
        ratio: reduction ratio for bottleneck channels
    Returns:
        output tensor of the SE block
    """
    channel_axis = -1 # TF default
    # Use get_shape().as_list() or shape property directly if in Eager mode,
    # K.int_shape might be less reliable sometimes depending on TF version/mode.
    # Let's try K.int_shape first as it's often used in older code.
    try:
      filters = K.int_shape(input_tensor)[channel_axis]
      if filters is None: # If shape inference failed
          filters = tf.shape(input_tensor)[channel_axis] # Fallback for dynamic shape
    except Exception:
       filters = tf.shape(input_tensor)[channel_axis] # More robust fallback

    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)

    # Excitation: Two Dense layers (FC layers) acting like 1x1 Convs
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # Reshape se for broadcasting to (Batch, 1, 1, C)
    se = layers.Reshape((1, 1, filters))(se)

    # Rescale (multiply) original input
    x = layers.multiply([input_tensor, se])
    return x

def build_model_var6(learning_rate=1e-3, dropout_rate=0.25, l2_reg=1.5e-3):
    """
    U-Net like architecture with skip connections and SE blocks. (FIXED with Cropping2D)
    Based on var5 filter counts to target ~25M MACs.
    Encoder Filters: 8 -> 16 -> 32 -> 64
    Decoder Filters (ConvT): 32 -> 16 -> 8
    Decoder Filters (Fusion): 32 -> 16 -> 8
    SE blocks added after bottleneck and decoder fusion stages.
    """
    # Default loss function using v2
    loss_fn = custom_obstacle_loss_v2(
        obstacle_weight=4.0,
        var_penalty_weight=0.001,
        diversity_penalty_weight=0.0001,
        alpha_grad=0.2
    )

    reg = regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(240, 520, 3))

    # --- Encoder ---
    # Stage 1
    e1 = layers.Conv2D(8, (7, 7), strides=(7, 7), padding='same', activation='relu', kernel_regularizer=reg)(inputs)
    e1 = layers.BatchNormalization()(e1)
    # Output shape ~ (Batch, 35, 75, 8) - Skip 1

    # Stage 2
    e2 = layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e1)
    e2 = layers.BatchNormalization()(e2)
    # Output shape ~ (Batch, 18, 38, 16) - Skip 2

    # Stage 3
    e3 = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e2)
    e3 = layers.BatchNormalization()(e3)
    # Output shape ~ (Batch, 9, 19, 32) - Skip 3

    # Bottleneck (Stage 4)
    b = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(e3)
    b = layers.BatchNormalization()(b)
    b = se_block(b) # Add SE block after bottleneck BN
    b = layers.Dropout(dropout_rate)(b) # Dropout after SE
    # Output shape ~ (Batch, 5, 10, 64)

    # --- Decoder ---
    # Stage 1 (Upsample + Skip 3)
    d1 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(b)
    d1 = layers.BatchNormalization()(d1)
    # Output shape of d1 is ~ (Batch, 10, 20, 32)

    # ---> FIX: Crop d1 to match spatial dimensions of e3 <---
    d1 = layers.Cropping2D(cropping=((1, 0), (1, 0)), name='crop_d1_to_e3_var6')(d1)

    # Concatenate with e3 (shape (9, 19))
    d1 = layers.Concatenate()([d1, e3])
    # Fusion Conv
    d1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = se_block(d1) # Add SE block after fusion BN
    d1 = layers.Dropout(dropout_rate)(d1) # Dropout after SE
    # Output shape ~ (Batch, 9, 19, 32)

    # Stage 2 (Upsample + Skip 2)
    d2 = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(d1)
    d2 = layers.BatchNormalization()(d2)
    # Output shape d2: ~ (Batch, 18, 38, 16) - Should match e2

    # Concatenate with e2
    d2 = layers.Concatenate()([d2, e2])
    # Fusion Conv
    d2 = layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = se_block(d2) # Add SE block after fusion BN
    d2 = layers.Dropout(dropout_rate)(d2) # Dropout after SE
    # Output shape ~ (Batch, 18, 38, 16)

    # Stage 3 (Upsample + Skip 1)
    d3 = layers.Conv2DTranspose(8, (3, 3), strides=2, padding='same', activation='relu', kernel_regularizer=reg)(d2)
    d3 = layers.BatchNormalization()(d3)
    # Output shape d3: ~ (Batch, 36, 76, 8) - Check against e1 (35, 75, 8) -> Mismatch!

    # ---> FIX: Crop d3 to match spatial dimensions of e1 <---
    d3 = layers.Cropping2D(cropping=((1, 0), (1, 0)), name='crop_d3_to_e1_var6')(d3)

    # Concatenate with e1
    d3 = layers.Concatenate()([d3, e1])
    # Fusion Conv
    d3 = layers.Conv2D(8, (3, 3), padding='same', activation='relu', kernel_regularizer=reg)(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = se_block(d3) # Add SE block after fusion BN
    d3 = layers.Dropout(dropout_rate / 1.5)(d3) # Dropout after SE (optional reduced rate)
    # Output shape ~ (Batch, 35, 75, 8)

    # --- Output Head ---
    output = layers.Conv2D(1, (3, 3), padding='same', activation=None)(d3)
    output = layers.Activation(scaled_sigmoid)(output)
    output = layers.Resizing(12, 32)(output)
    output = layers.Flatten()(output)

    model = models.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    return model

def build_optimized_model(architecture, learning_rate, dropout_rate, l2_reg,
                          obstacle_weight, var_penalty_weight, diversity_penalty_weight,
                          alpha_grad):
    """Build model with the specified architecture and optimized hyperparameters"""
    loss_fn = custom_obstacle_loss_v2(obstacle_weight, var_penalty_weight,
                                      diversity_penalty_weight, alpha_grad)
    
    if architecture == 'base':
        model = build_model_base(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var1':
        model = build_model_var1(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var2':
        model = build_model_var2(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var3':
        model = build_model_var3(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var4':
        model = build_model_var4(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var5':
        model = build_model_var5(learning_rate, dropout_rate, l2_reg)
    elif architecture == 'var6':
        model = build_model_var6(learning_rate, dropout_rate, l2_reg)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', f1_score_metric, ssim_metric, intra_pred_variation, inter_pred_variation]
    )
    
    return model

###############################
# Data Augmentation
###############################
def rotate_image_tf(image, angle):
    """Rotate image using TF projective transform"""
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx = w / 2.0
    cy = h / 2.0
    cos_val = tf.cos(angle)
    sin_val = tf.sin(angle)
    a = cos_val
    b = -sin_val
    c = cx - cos_val * cx + sin_val * cy
    d = sin_val
    e = cos_val
    f = cy - sin_val * cx - cos_val * cy
    transform = [a, b, c, d, e, f, 0.0, 0.0]
    image_batch = tf.expand_dims(image, 0)
    transforms = tf.convert_to_tensor([transform], dtype=tf.float32)
    output_shape = tf.convert_to_tensor([tf.shape(image)[0], tf.shape(image)[1]], dtype=tf.int32)
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_batch,
        transforms=transforms,
        output_shape=output_shape,
        interpolation="BILINEAR",
        fill_value=0
    )
    return tf.squeeze(transformed, axis=0)

def gaussian_blur_tf(image, sigma):
    """Apply Gaussian blur using depthwise convolution"""
    kernel_size = tf.cast(2 * tf.math.ceil(2 * sigma) + 1, tf.int32)
    x = tf.range(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=tf.float32)
    gauss = tf.exp(- (x**2) / (2.0 * sigma**2))
    gauss /= tf.reduce_sum(gauss)
    gauss = tf.reshape(gauss, [kernel_size, 1])
    kernel = gauss * tf.transpose(gauss)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel, [1, 1, channels, 1])
    image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(blurred, axis=0)

@tf.function
def zoom_fn(image):
    """Apply zoom augmentation"""
    zoom_factor = tf.random.uniform([], 0.7, 1.3)
    orig_shape = tf.shape(image)[:2]
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * zoom_factor, tf.int32)
    image_resized = tf.image.resize(image, new_size)
    
    def zoom_in():
        offset = (new_size - orig_shape) // 2
        return tf.image.crop_to_bounding_box(image_resized, offset[0], offset[1],
                                             orig_shape[0], orig_shape[1])
    
    def zoom_out():
        pad_total = orig_shape - new_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padded = tf.pad(image_resized, [[pad_before[0], pad_after[0]], [pad_before[1], pad_after[1]], [0, 0]], mode='CONSTANT')
        return tf.image.resize_with_crop_or_pad(padded, orig_shape[0], orig_shape[1])
    
    return tf.cond(zoom_factor > 1.0, zoom_in, zoom_out)

@tf.function
def augment_single_image(image, label):
    """Apply GPU-friendly augmentations to a single image and its label"""
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    
    def augment_body():
        local_image = image / 255.0
        local_label = label
        
        do_flip = tf.less(tf.random.uniform([]), 0.5)
        local_image = tf.cond(do_flip, lambda: tf.image.flip_left_right(local_image), lambda: local_image)
        local_label = tf.cond(do_flip, lambda: tf.reverse(local_label, axis=[1]), lambda: local_label)
        
        brightness_delta = tf.random.uniform([], -0.3, 0.3)
        local_image = tf.cond(
            tf.random.uniform([]) < 0.2,
            lambda: tf.clip_by_value(tf.image.adjust_brightness(local_image, brightness_delta), 0.0, 1.0),
            lambda: local_image
        )
        
        contrast_factor = tf.random.uniform([], 0.7, 1.3)
        local_image = tf.cond(
            tf.random.uniform([]) < 0.2,
            lambda: tf.clip_by_value(tf.image.adjust_contrast(local_image, contrast_factor), 0.0, 1.0),
            lambda: local_image
        )
        
        gamma_factor = tf.random.uniform([], 0.7, 1.3)
        local_image = tf.cond(
            tf.random.uniform([]) < 0.2,
            lambda: tf.clip_by_value(tf.image.adjust_gamma(local_image, gamma_factor), 0.0, 1.0),
            lambda: local_image
        )

        local_image = tf.cond(
            tf.random.uniform([]) < 0.05,
            lambda: gaussian_blur_tf(local_image, tf.random.uniform([], 1.0, 2.0)),
            lambda: local_image
        )
        
        local_image = tf.cond(
            tf.random.uniform([]) < 0.2,
            lambda: tf.cond(
                tf.random.uniform([]) < 0.5,
                lambda: tf.clip_by_value(
                    local_image + tf.random.normal(tf.shape(local_image), 0.0, 0.1), 0.0, 1.0
                ),
                lambda: tf.where(
                    tf.expand_dims(tf.less(tf.random.uniform(tf.shape(local_image)[:2]), 0.03), -1),
                    1.0, local_image
                )
            ),
            lambda: local_image
        )
        
        local_image = tf.cond(
            tf.random.uniform([]) < 0.35,
            lambda: tf.image.adjust_hue(local_image, tf.random.uniform([], -0.5, 0.5)),
            lambda: local_image
        )
        
        local_image = tf.cond(
            tf.random.uniform([]) < 0.35,
            lambda: tf.image.adjust_saturation(local_image, tf.random.uniform([], 0.6, 1.4)),
            lambda: local_image
        )
        
        local_image = local_image * 255.0
        return local_image, local_label

    image, label = tf.cond(tf.random.uniform([]) < 0.2,
                           lambda: (image, label),
                           lambda: augment_body())
    
    image = tf.clip_by_value(image, 0.0, 255.0)
    label = tf.clip_by_value(label, 0.0, 255.0)
    return image, label

def create_augmented_dataset(x, y, batch_size=32, shuffle_buffer=1000, augment=True):
    """Create a tf.data.Dataset with augmentation and preprocessing"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    if augment:
        dataset = dataset.map(augment_single_image, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(
            lambda img, lbl: (tf.cast(img, tf.float32), tf.cast(lbl, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    fixed_shape = x.shape[1:]
    dataset = dataset.map(
        lambda img, lbl: (img, tf.reshape(lbl, (384,))),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

###############################
# Hyperparameter Optimization
###############################
# Updated search space
ARCHITECTURE = Categorical(['var1', 'var2', 'var3', 'var4', 'var5'], name='architecture')
VAR_PENALTY_WEIGHT = Real(5e-4, 0.1, prior='log-uniform', name='var_penalty_weight') # Was 1e-4
DIVERSITY_PENALTY_WEIGHT = Real(5e-5, 0.05, prior='log-uniform', name='diversity_penalty_weight') # Was 1e-5
ALPHA_GRAD = Real(0.1, 0.7, name='alpha_grad') # Was 0.35, 0.6
LEARNING_RATE = Real(1e-6, 1e-3, prior='log-uniform', name='learning_rate')
DROPOUT_RATE = Real(0.1, 0.55, name='dropout_rate')
L2_REG = Real(1e-4, 1e-2, prior='log-uniform', name='l2_reg')
BATCH_SIZE = Categorical([8, 16, 32, 64], name='batch_size')
OBSTACLE_WEIGHT = Real(1.5, 8.0, name='obstacle_weight')

search_space = [
    ARCHITECTURE,
    OBSTACLE_WEIGHT,
    VAR_PENALTY_WEIGHT,
    DIVERSITY_PENALTY_WEIGHT,
    ALPHA_GRAD,
    LEARNING_RATE,
    DROPOUT_RATE,
    L2_REG,
    BATCH_SIZE
]

stop_optimization = False

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop optimization"""
    global stop_optimization
    print("\nOptimization stopping after current iteration completes...")
    stop_optimization = True

signal.signal(signal.SIGINT, signal_handler)

def save_training_plot(history, trial_id, params):
    """Save a plot of training and validation metrics"""
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Trial {trial_id} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(history['mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.title(f'Trial {trial_id} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    if 'f1_score_metric' in history:
         plt.plot(history['f1_score_metric'], label='Train F1')
    if 'val_f1_score_metric' in history:
         plt.plot(history['val_f1_score_metric'], label='Val F1')
    if 'ssim_metric' in history:
         plt.plot(history['ssim_metric'], label='Train SSIM')
    if 'val_ssim_metric' in history:
         plt.plot(history['val_ssim_metric'], label='Val SSIM')
    plt.title(f'Trial {trial_id} - F1 & SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    
    bo_val = history.get('bo_objective_value', history['val_mae'][-1])
    filename = f"trial_{trial_id}_lr_{params['learning_rate']:.2e}_bo_{bo_val:.4f}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def log_results(trial_id, params, history, duration):
    """Log trial results to CSV and console"""
    final_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_mae = history['mae'][-1]
    final_val_mae = history['val_mae'][-1]
    epochs_run = len(history['loss'])
    
    best_val_f1 = 0.0
    if 'val_f1_score_metric' in history:
         best_val_f1 = max(history['val_f1_score_metric'])
    best_val_ssim = 0.0
    if 'val_ssim_metric' in history:
         best_val_ssim = max(history['val_ssim_metric'])
    
    w_f1 = 0.5
    w_ssim = 0.5
    bo_objective_value = w_f1 * (1.0 - best_val_f1) + w_ssim * (1.0 - best_val_ssim)
    
    history['bo_objective_value'] = bo_objective_value
    
    result = {
        'trial_id': trial_id,
        'architecture': params['architecture'],
        'obstacle_weight': params['obstacle_weight'],
        'var_penalty_weight': params['var_penalty_weight'],
        'diversity_penalty_weight': params['diversity_penalty_weight'],
        'alpha_grad': params['alpha_grad'],
        'learning_rate': params['learning_rate'],
        'dropout_rate': params['dropout_rate'],
        'l2_reg': params['l2_reg'],
        'batch_size': params['batch_size'],
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'final_mae': final_mae,
        'final_val_mae': final_val_mae,
        'epochs': epochs_run,
        'duration_minutes': duration / 60,
        'best_val_f1': best_val_f1,
        'best_val_ssim': best_val_ssim,
        'bo_objective_value': bo_objective_value
    }
    
    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result])
    
    results_df.to_csv(RESULTS_CSV, index=False)
    
    print(f"\n===== TRIAL {trial_id} RESULTS =====")
    print(f"Duration: {duration/60:.2f} minutes")
    print(f"Architecture: {params['architecture']}")
    print(f"Loss Params: Obstacle Weight={params['obstacle_weight']:.2f}, Grad={params['alpha_grad']:.3f}")
    print(f"Penalty Weights: Var={params['var_penalty_weight']:.4f}, Div={params['diversity_penalty_weight']:.5f}")
    print(f"Final metrics: Val Loss={final_val_loss:.4f}, Val MAE={final_val_mae:.4f}")
    print(f"Best Val F1: {best_val_f1:.4f}, Best Val SSIM: {best_val_ssim:.4f}")
    print(f"BO Objective Value: {bo_objective_value:.4f}")
    print(f"Results saved to {RESULTS_CSV}")
    
    return bo_objective_value

def objective_impl(architecture, var_penalty_weight, diversity_penalty_weight, alpha_grad,
                   learning_rate, dropout_rate, l2_reg, batch_size, obstacle_weight):
    """Implementation of the objective function that takes named parameters"""
    global trial_counter
    
    print(f"\n===== STARTING TRIAL {trial_counter} =====")
    params = {
        'architecture': architecture,
        'var_penalty_weight': var_penalty_weight,
        'diversity_penalty_weight': diversity_penalty_weight,
        'alpha_grad': alpha_grad,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
        'batch_size': int(batch_size),
        'obstacle_weight': obstacle_weight
    }
    
    for param, value in params.items():
        print(f"{param}: {value}")
    
    model = build_optimized_model(
        architecture=architecture,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        obstacle_weight=obstacle_weight,
        var_penalty_weight=var_penalty_weight,
        diversity_penalty_weight=diversity_penalty_weight,
        alpha_grad=alpha_grad
    )
    
    train_ds = create_augmented_dataset(X_TRAIN, y_TRAIN, batch_size=int(batch_size), 
                                      shuffle_buffer=2000, augment=True)
    val_ds = create_augmented_dataset(X_TEST, y_TEST, batch_size=int(batch_size), 
                                    shuffle_buffer=0, augment=False)
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score_metric', factor=0.5, patience=5, verbose=1, min_lr=1e-7, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_metric', patience=10, restore_best_weights=True, mode='max')
    ]
    
    start_time = time.time()
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=250,
        callbacks=callbacks,
        verbose=2
    ).history
    duration = time.time() - start_time
    
    model_path = os.path.join(MODELS_DIR, f"trial_{trial_counter}_model.h5")
    model.save(model_path)
    
    bo_objective = log_results(trial_counter, params, history, duration)
    save_training_plot(history, trial_counter, params)
    
    with open(os.path.join(RESULTS_DIR, f"trial_{trial_counter}_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    
    trial_counter += 1
    
    return bo_objective

@use_named_args(search_space)
def objective(*args, **kwargs):
    """Wrapper for the objective function that works with both direct calls and decorated calls"""
    if len(args) == 1 and isinstance(args[0], list):
        params_list = args[0]
        params_dict = {dim.name: val for dim, val in zip(search_space, params_list)}
        return objective_impl(**params_dict)
    elif len(args) == len(search_space) and not kwargs:
        params_dict = {dim.name: val for dim, val in zip(search_space, args)}
        return objective_impl(**params_dict)
    else:
        return objective_impl(**kwargs)

def run_random_search(n_iterations=30, target_bo_objective=0.15):
    """Run random search until reaching target BO objective or completing iterations"""
    global trial_counter
    best_bo_objective = float('inf')
    best_params = None
    all_points = []
    all_values = []
    
    print(f"\n===== STARTING RANDOM SEARCH PHASE (Target BO Objective: {target_bo_objective}) =====\n")
    
    for i in range(n_iterations):
        if stop_optimization:
            print("Random search stopped by user.")
            break
            
        random_params = {}
        for param in search_space:
            if isinstance(param, Real):
                random_params[param.name] = np.random.uniform(param.low, param.high)
                if param.prior == 'log-uniform':
                    log_low = np.log10(param.low)
                    log_high = np.log10(param.high)
                    random_params[param.name] = 10 ** np.random.uniform(log_low, log_high)
            elif isinstance(param, Integer):
                random_params[param.name] = np.random.randint(param.low, param.high + 1)
            elif isinstance(param, Categorical):
                random_params[param.name] = np.random.choice(param.categories)
        
        try:
            bo_objective = objective_impl(**random_params)
            
            x_point = [random_params[param.name] for param in search_space]
            all_points.append(x_point)
            all_values.append(bo_objective)
            
            if bo_objective < best_bo_objective:
                best_bo_objective = bo_objective
                best_params = random_params
                
            if bo_objective <= target_bo_objective:
                print(f"\n===== RANDOM SEARCH ACHIEVED TARGET BO Objective ({bo_objective:.4f} <= {target_bo_objective}) =====")
                break
        except Exception as e:
            print(f"Error during trial {trial_counter}: {e}")
            print("Parameters that caused the error:", random_params)
            raise
    
    print(f"\n===== RANDOM SEARCH COMPLETED =====")
    print(f"Best BO Objective: {best_bo_objective:.4f}")
    print(f"Random search evaluations: {len(all_points)}")
    
    return best_params, best_bo_objective, all_points, all_values

def run_bayesian_optimization(n_iterations=500, initial_points=None, x0=None, y0=None):
    """Run Bayesian optimization with warm start from previous evaluations"""
    global trial_counter
    
    print(f"\n===== STARTING BAYESIAN OPTIMIZATION PHASE =====\n")
    print(f"Starting with {len(x0) if x0 else 0} pre-evaluated points")
    
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_iterations,
        x0=x0,
        y0=y0,
        n_initial_points=max(10 - len(x0) if x0 else 10, 0),
        verbose=True,
        callback=lambda res: check_stop_flag(res)
    )
    
    best_params = {param.name: value for param, value in zip(search_space, result.x)}
    
    with open(os.path.join(RESULTS_DIR, "bayesian_opt_result.pkl"), "wb") as f:
        pickle.dump(result, f)
    
    plot_fig = plot_convergence(result)
    plot_fig.savefig(os.path.join(RESULTS_DIR, "convergence_plot.png"))
    
    print(f"\n===== BAYESIAN OPTIMIZATION COMPLETED =====")
    print(f"Best BO Objective: {result.fun:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, result.fun

def check_stop_flag(res):
    """Check if optimization should be stopped"""
    return stop_optimization

def find_most_recent_results_dir():
    """Find the most recent results directory for warm start"""
    base_dir = "hyperopt_results"
    if not os.path.exists(base_dir):
        raise ValueError(f"No previous results found in {base_dir}. Cannot use warm start.")
    
    dirs = glob.glob(os.path.join(base_dir, "????????_??????"))
    if not dirs:
        raise ValueError(f"No timestamp directories found in {base_dir}. Cannot use warm start.")
    
    dirs.sort()
    most_recent_dir = dirs[-1]
    
    results_csv = os.path.join(most_recent_dir, "optimization_results.csv")
    if not os.path.exists(results_csv):
        raise ValueError(f"Results CSV not found in {most_recent_dir}. Cannot use warm start.")
    
    print(f"Using most recent results from: {most_recent_dir}")
    return most_recent_dir

def load_previous_evaluations(results_dir):
    """Load previous evaluations from results CSV for warm start"""
    results_csv = os.path.join(results_dir, "optimization_results.csv")
    results_df = pd.read_csv(results_csv)
    
    print(f"Loaded {len(results_df)} previous evaluations for warm start")
    
    x0 = []
    y0 = []
    
    for _, row in results_df.iterrows():
        point = []
        for dim in search_space:
            if dim.name in row:
                point.append(row[dim.name])
            else:
                if dim.name == 'obstacle_weight':
                    point.append(4.0)
                elif isinstance(dim, Categorical):
                    point.append(dim.categories[0])
                elif isinstance(dim, Real) or isinstance(dim, Integer):
                    point.append(dim.low)
        if len(point) == len(search_space):
            x0.append(point)
            y0.append(row['bo_objective_value'])
    
    print(f"Converted {len(x0)} evaluations to format suitable for warm start")
    return x0, y0

###############################
# Main Function
###############################
def main():
    """Main function to run the hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Train obstacle avoidance CNN with hyperparameter optimization')
    parser.add_argument('--warm', action='store_true', help='Skip random search and continue with Bayesian optimization from previous results')
    args = parser.parse_args()
    
    global RESULTS_DIR, RESULTS_CSV, MODELS_DIR, PLOTS_DIR
    if args.warm:
        RESULTS_DIR = find_most_recent_results_dir()
    else:
        TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR = os.path.join("hyperopt_results", TIMESTAMP)
        
    RESULTS_CSV = os.path.join(RESULTS_DIR, "optimization_results.csv")
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load data
    labels_data = {}
    with open(LABELS_JSON_PROVIDEDIMAGES, "r") as f:
        labels_data["provided_images"] = json.load(f)
    with open(LABELS_JSON_VIDEOCAPSIMULATIONROUND1, "r") as f:
        labels_data["videocap_simulation_round1"] = json.load(f)
    with open(LABELS_JSON_VIDEOCAPSIMULATIONROUND2, "r") as f:
        labels_data["videocap_simulation_round2"] = json.load(f)

    # Process test_round2: reorder temporally and split into two parts
    with open(LABELS_JSON_TESTROUND2, "r") as f:
        test2_data = json.load(f)

    # Sort the keys (which represent indices) in ascending order (early to late)
    sorted_keys = sorted(test2_data.keys(), key=lambda k: int(k))
    ordered_test2_data = {k: test2_data[k] for k in sorted_keys}

    # Split: the last 1000 images for test, the rest for training
    num_test = 1000
    train_keys = sorted_keys[:-num_test]
    test_keys = sorted_keys[-num_test:]
    train_test2_data = {k: ordered_test2_data[k] for k in train_keys}
    test_test2_data = {k: ordered_test2_data[k] for k in test_keys}

    X_VIDEOCAP1, y_VIDEOCAP1 = load_data(
        IMAGE_FOLDER_VIDEOCAPSIMULATIONROUND1, 
        labels_data["videocap_simulation_round1"]
    )
    X_VIDEOCAP2, y_VIDEOCAP2 = load_data(
        IMAGE_FOLDER_VIDEOCAPSIMULATIONROUND2, 
        labels_data["videocap_simulation_round2"]
    )
    X_TEST2_train, y_TEST2_train = load_data(
        IMAGE_FOLDER_TESTROUND2, 
        train_test2_data
    )
    X_TEST2_test, y_TEST2_test = load_data(
        IMAGE_FOLDER_TESTROUND2, 
        test_test2_data
    )
    X_PROVIDED, y_PROVIDED = load_data(
        IMAGE_FOLDER_PROVIDEDIMAGES, 
        labels_data["provided_images"]
    )

    # Build training set using provided images and test_round2's train part
    X_TRAIN_INITIAL = np.concatenate([X_VIDEOCAP1, X_VIDEOCAP2, X_PROVIDED, X_TEST2_train], axis=0)
    y_TRAIN_INITIAL = np.concatenate([y_VIDEOCAP1, y_VIDEOCAP2, y_PROVIDED, y_TEST2_train], axis=0)

    np.random.seed(43)
    num_samples_to_move = int(0 * len(X_TRAIN_INITIAL))
    indices_to_move = np.random.choice(len(X_TRAIN_INITIAL), num_samples_to_move, replace=False)
    mask = np.ones(len(X_TRAIN_INITIAL), dtype=bool)
    mask[indices_to_move] = False

    global X_TRAIN, y_TRAIN, X_TEST, y_TEST
    X_TRAIN = X_TRAIN_INITIAL[mask]
    y_TRAIN = y_TRAIN_INITIAL[mask]

    # Test set is now solely from test_round2's test part (1000 images)
    X_TEST = X_TEST2_test
    y_TEST = y_TEST2_test

    print(f"Train set: {len(X_TRAIN)} images, Test set: {len(X_TEST)} images")

        
    global trial_counter
    if args.warm and os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        trial_counter = results_df['trial_id'].max() + 1 if not results_df.empty else 1
        print(f"Continuing from trial {trial_counter}")
    else:
        trial_counter = 1
    
    print("Starting hyperparameter optimization with multiple architectures...")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    try:
        if args.warm:
            print("\n===== WARM START: LOADING PREVIOUS EVALUATIONS =====")
            x0, y0 = load_previous_evaluations(RESULTS_DIR)
            
            if not x0:
                print("No valid previous evaluations found. Cannot use warm start.")
                return
            
            best_idx = np.argmin(y0)
            best_params = {dim.name: x0[best_idx][i] for i, dim in enumerate(search_space)}
            best_mae = y0[best_idx]
            print(f"Best BO Objective from previous evaluations: {best_mae:.4f}")
            
            best_params, best_mae = run_bayesian_optimization(
                n_iterations=500,
                initial_points=best_params,
                x0=x0,
                y0=y0
            )
        else:
            best_random_params, best_random_mae, rgs_points, rgs_values = run_random_search(n_iterations=30, target_bo_objective=0.15)
            
            if not stop_optimization:
                best_params, best_mae = run_bayesian_optimization(
                    n_iterations=500,
                    initial_points=best_random_params,
                    x0=rgs_points,
                    y0=rgs_values
                )
        
        if not stop_optimization:
            print("\n===== TRAINING FINAL MODEL WITH BEST PARAMETERS =====")
            final_model = build_optimized_model(
                architecture=best_params['architecture'],
                learning_rate=best_params['learning_rate'],
                dropout_rate=best_params['dropout_rate'],
                l2_reg=best_params['l2_reg'],
                obstacle_weight=best_params['obstacle_weight'],
                var_penalty_weight=best_params['var_penalty_weight'],
                diversity_penalty_weight=best_params['diversity_penalty_weight'],
                alpha_grad=best_params['alpha_grad']
            )
            
            final_batch_size = int(best_params['batch_size'])
            train_ds = create_augmented_dataset(X_TRAIN, y_TRAIN, batch_size=final_batch_size, 
                                              shuffle_buffer=2000, augment=True)
            val_ds = create_augmented_dataset(X_TEST, y_TEST, batch_size=final_batch_size, 
                                            shuffle_buffer=0, augment=False)
            
            final_history = final_model.fit(
                train_ds, 
                validation_data=val_ds, 
                epochs=1000,
                callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score_metric', factor=0.5, patience=8, verbose=1, mode='max'),
                    tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_metric', patience=15, restore_best_weights=True, mode='max')
                ],
                verbose=1
            ).history
            
            final_model_path = os.path.join(RESULTS_DIR, "final_model.h5")
            final_model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
            
            save_training_plot(final_history, "final", best_params)
            
    except KeyboardInterrupt:
        print("\nOptimization manually interrupted.")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(RESULTS_CSV):
            results_df = pd.read_csv(RESULTS_CSV)
            summary_path = os.path.join(RESULTS_DIR, "optimization_summary.txt")
            
            with open(summary_path, "w") as f:
                f.write(f"Hyperparameter Optimization Summary\n")
                f.write(f"================================\n\n")
                f.write(f"Total trials: {results_df.shape[0]}\n")
                
                arch_counts = results_df['architecture'].value_counts()
                f.write("\nArchitecture Distribution:\n")
                for arch, count in arch_counts.items():
                    f.write(f"  {arch}: {count} trials\n")
                
                best_idx = results_df['bo_objective_value'].idxmin()
                best_trial = results_df.iloc[best_idx]
                
                f.write(f"\nBest Trial ({best_trial['trial_id']}):\n")
                f.write(f"  Architecture: {best_trial['architecture']}\n")
                f.write(f"  BO Objective Value: {best_trial['bo_objective_value']:.4f}\n")
                f.write(f"  Best Val F1: {best_trial['best_val_f1']:.4f}\n")
                f.write(f"  Best Val SSIM: {best_trial['best_val_ssim']:.4f}\n")
                f.write(f"  Val Loss: {best_trial['final_val_loss']:.4f}\n")
                f.write(f"  Loss Params:\n")
                for param in search_space:
                    if param.name != 'architecture':
                        f.write(f"    {param.name}: {best_trial[param.name]}\n")
                
                f.write("\nBest Result Per Architecture:\n")
                for arch in arch_counts.index:
                    arch_df = results_df[results_df['architecture'] == arch]
                    if not arch_df.empty:
                        best_arch_idx = arch_df['bo_objective_value'].idxmin()
                        best_arch_trial = arch_df.iloc[best_arch_idx]
                        f.write(f"  {arch}: Trial {best_arch_trial['trial_id']}, BO Objective = {best_arch_trial['bo_objective_value']:.4f}\n")
                
            print(f"\nOptimization complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
