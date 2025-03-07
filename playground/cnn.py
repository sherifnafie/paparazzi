import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Import train_test_split

# Paths
IMAGE_FOLDER = r"C:\Users\beren\Documents\paparazzi\playground\videocap_simulation_round1"
LABELS_JSON = r"C:\Users\beren\Documents\paparazzi\playground\videocap_simulation_round1_labels\results.json"
OUTPUT_FOLDER = r"C:\Users\beren\Documents\paparazzi\playground\cnn_ouput_images"
VIDEO_OUTPUT = r"C:\Users\beren\Documents\paparazzi\playground\predictions.mp4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load labels
with open(LABELS_JSON, "r") as f:
    labels_data = json.load(f)

# Load images and labels
def load_data(image_folder, labels_data, img_size=(520, 240)):
    images, labels = [], []
    for filename, data in labels_data.items():
        img_path = os.path.join(image_folder, f"{filename}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        images.append(img)
        
        label_grid = np.array(data["scores"])  # Safety grid
        labels.append(label_grid)
    
    return np.array(images), np.array(labels)

X, y = load_data(IMAGE_FOLDER, labels_data)
y = np.expand_dims(y, axis=-1)  # Add channel for CNN

# Randomly split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# CNN Model
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(y_train.shape[1] * y_train.shape[2], activation='sigmoid'),  # Output size of safety grid
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train.shape[1:])

# EarlyStopping callback setup
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model with EarlyStopping
model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=5, batch_size=16, 
          validation_data=(X_test, y_test.reshape(y_test.shape[0], -1)),
          callbacks=[early_stopping])  # Add EarlyStopping callback

# Predict on test set
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(y_test.shape)  # Reshape to grid format

def overlay_safety_grid(base_img, safety_grid, alpha=0.4):
    """
    Overlays the safety grid onto a base image.
    Grid values in [0,1] are mapped: 0 -> red (unsafe) and 1 -> green (safe).
    """
    overlay = base_img.copy()
    h, w, _ = overlay.shape
    grid_height, grid_width, extra = safety_grid.shape
    cell_h = h // grid_height
    cell_w = w // grid_width

    for gy in range(grid_height):
        for gx in range(grid_width):
            val = float(safety_grid[gy, gx])  # Ensure it's a scalar
            r = int((1.0 - val) * 255)
            g = int(val * 255)
            b = 0
            color = (b, g, r)  # OpenCV uses BGR
            
            y_start = gy * cell_h
            y_end   = (gy+1) * cell_h if gy < grid_height - 1 else h
            x_start = gx * cell_w
            x_end   = (gx+1) * cell_w if gx < grid_width - 1 else w

            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)

    blended = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)
    return blended

# Save predictions
def save_predicted_images(X_test, y_pred, y_test, output_folder):
    for i in range(len(X_test)):
        original = (X_test[i] * 255).astype(np.uint8)
        pred_overlay = overlay_safety_grid(original, y_pred[i])
        gt_overlay = overlay_safety_grid(original, y_test[i])
        
        combined = np.hstack((gt_overlay, pred_overlay))  # Side-by-side comparison
        cv2.imwrite(os.path.join(output_folder, f"pred_{i}.jpg"), combined)

save_predicted_images(X_test, y_pred, y_test, OUTPUT_FOLDER)

# Generate video
def create_video(image_folder, output_video, fps=5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    
    if not images:
        print("No images found to create video.")
        return
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)
    
    video.release()
    print(f"Video saved to {output_video}")

create_video(OUTPUT_FOLDER, VIDEO_OUTPUT)

model.save("cnn_model")  # Saves the model in TensorFlow SavedModel format in current working library

import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model")
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
with open("cnn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
