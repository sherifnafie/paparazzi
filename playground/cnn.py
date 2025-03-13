import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Import train_test_split
import tf2onnx
import onnx

# Paths
IMAGE_FOLDER = r"C:\Users\beren\Documents\paparazzi\playground\videocap_simulation_round1"
LABELS_JSON = r"C:\Users\beren\Documents\paparazzi\playground\videocap_simulation_round1_labels\results.json"
OUTPUT_FOLDER = r"C:\Users\beren\Documents\paparazzi\playground\cnn_ouput_images"
VIDEO_OUTPUT = r"C:\Users\beren\Documents\paparazzi\playground\predictions.mp4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load labels
with open(LABELS_JSON, "r") as f:
    labels_data = json.load(f)

# Load images and labels, converting to YUV color space
def load_data(image_folder, labels_data, img_size=(520, 240)):
    images, labels = [], []
    i = 0
    for filename, data in labels_data.items():
        img_path = os.path.join(image_folder, f"{filename}.jpg")
        if not os.path.exists(img_path):
            continue
        
        # Read the image
        img = cv2.imread(img_path)
        
        # Convert the image from BGR to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert to YUV
        
        # Append the image to the images list
        images.append(img_yuv)

        if i == 0:
            # Save the entire YUV image grid (each pixel with [Y, U, V] values) of the first image to a text file
            yuv_output_path = os.path.join(OUTPUT_FOLDER, "first_image_yuv_grid.txt")
            with open(yuv_output_path, "w") as f:
                # Write the YUV array as 3D grid
                np.savetxt(f, img_yuv.reshape((-1, 3)), fmt="%d")  # Flatten the image into a list of [Y, U, V] tuples
            print(f"YUV image grid of first image saved to {yuv_output_path}")
        
        # Get the label grid
        label_grid = np.array(data["scores"])  # Safety grid
        labels.append(label_grid)
        
        i += 1
    
    return np.array(images), np.array(labels)

X, y = load_data(IMAGE_FOLDER, labels_data)
y = np.expand_dims(y, axis=-1)  # Add channel for CNN

# Split the data into training and testing based on the index
num_images = len(X)
halfway_point = num_images // 2

X_train = X[:halfway_point]
y_train = y[:halfway_point]

X_test = X[halfway_point:]
y_test = y[halfway_point:]

# CNN Model with output layer named explicitly
def build_model(input_shape):
    model = models.Sequential([ 
        # Conv1: First convolution layer with larger strides and 1x1 kernel
        layers.Conv2D(16, (1, 1), strides=(40, 32), activation='relu', input_shape=input_shape),  # Higher stride
        layers.MaxPooling2D((2, 2)),  # Optional pooling after Conv1
        
        # Conv2: Second convolution layer with smaller filters
        layers.Conv2D(58, (1, 1), strides=(1, 1), activation='relu'),  # Smaller stride
        layers.MaxPooling2D((2, 2)),  # Optional pooling after Conv2
        
        # Conv3: Final convolution layer, channel compression
        layers.Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid'),  # Output to match the final grid
        layers.Flatten(),  # Flatten the final output
        
        # Dense layer for final grid output (flattened)
        layers.Dense(y_train.shape[1] * y_train.shape[2], activation='sigmoid', name='output')  # Named output layer
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

model.output_names=['output']

# Save the model
model.save(r"C:\Users\beren\Documents\paparazzi\playground\cnn_model.keras")

# Convert the trained model to ONNX format using tf2onnx
onnx_model_path = r"C:\Users\beren\Documents\paparazzi\playground\cnn_model.onnx"

# Convert the model to ONNX format with an explicit output name
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    opset=13,
    input_signature=[tf.TensorSpec(shape=[None, 240, 520, 3], dtype=tf.float32)],
)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX format!")

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Find the first image in the folder
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")])
if not image_files:
    print("No images found in the folder.")
else:
    first_image_path = os.path.join(IMAGE_FOLDER, image_files[0])

    # Load and preprocess the image
    first_img = cv2.imread(first_image_path)
    first_img_yuv = cv2.cvtColor(first_img, cv2.COLOR_BGR2YUV)  # Convert to YUV
    first_img_yuv = np.expand_dims(first_img_yuv, axis=0)  # Add batch dimension

    # Predict
    first_pred = model.predict(first_img_yuv)
    first_pred = first_pred.reshape(y_train.shape[1:])  # Reshape to match output grid

    # Save the safety grid of the first image in a text file
    safety_grid_output_path = os.path.join(OUTPUT_FOLDER, "first_image_safety_grid.txt")
    with open(safety_grid_output_path, "w") as f:
        for row in first_pred[0]:  # Iterate through the predicted safety grid
            f.write(" ".join([f"{val:.4f}" for val in row]) + "\n")
    
    print(f"Safety grid of the first image saved to {safety_grid_output_path}")

    # Overlay safety grid
    pred_overlay = overlay_safety_grid(first_img, first_pred)

    # Save or show the result
    output_first_pred_path = os.path.join(OUTPUT_FOLDER, "first_image_prediction.jpg")
    cv2.imwrite(output_first_pred_path, pred_overlay)
    print(f"Prediction on first image saved to {output_first_pred_path}")

    # Optionally display the image
    plt.imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Safety Grid on First Image")
    plt.axis("off")
    plt.show()
