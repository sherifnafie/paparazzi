# Import necessary libraries
import os
import cv2
import json
import numpy as np
import tensorflow as tf
# --- Debugging Additions ---
import matplotlib # Add this line
matplotlib.use('Agg') # Force Agg backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import gc # Import garbage collector
from memory_profiler import profile # Import memory profiler decorator
# --- End Debugging Additions ---
import random
import sys
import argparse
import io # Keep io for buffer rendering

# Parse command line arguments first
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint model or ONNX model')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX model instead of TensorFlow model')
    parser.add_argument('--single', type=str, help='Path to a single image for evaluation')
    parser.add_argument('--model', type=str, help='Path to the specific model file (TF .h5 or .onnx)')
    parser.add_argument('--video', action='store_true', help='Process all images and save output as a video')
    # Keep args for data/labels but use original defaults if not provided
    parser.add_argument('--data', type=str, default="videocap_testround2", help='Name of the dataset folder to use (default: videocap_testround2)')
    parser.add_argument('--labels', type=str, default="labelled_videocap_testround2/results.json", help='Path to the corresponding labels JSON file (default: labelled_videocap_testround2/results.json)')
    # Removed --output arg as we are reverting to hardcoded defaults
    return parser.parse_args()

args = parse_args()

# Check for ONNX Runtime only if --onnx flag is used
if args.onnx:
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
    except ImportError:
        print("Error: ONNX Runtime not installed but --onnx flag was used.")
        print("Please install ONNX Runtime using:")
        print("  pip install onnxruntime")
        # Update path if necessary
        # print("\nAlternatively, run the installation script:")
        # print("  bash /path/to/your/install_onnxruntime.sh")
        sys.exit(1)

# Set paths and directories based on arguments (using defaults if not provided)
IMAGE_FOLDER = args.data
LABELS_JSON = args.labels

# Determine model path
if args.model:
    MODEL_PATH = args.model
elif args.onnx:
    MODEL_PATH = "cnn_model.onnx" # Default ONNX name
else:
    # Default TF model path - **Make sure this is updated or use --model**
    MODEL_PATH = r'/home/lievijn/training_share/hyperopt_results/20250328_003714/models/trial_29_model.h5'
    print(f"Warning: Using default TF model path: {MODEL_PATH}. Use --model to specify.")

# *** Reverted OUTPUT_FOLDER logic to original defaults ***
OUTPUT_FOLDER = "evaluation_results_onnx" if args.onnx else "evaluation_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# *** End Reverted Section ***

print(f"--- Configuration ---")
print(f"Evaluating Model: {MODEL_PATH}")
print(f"Model Type: {'ONNX' if args.onnx else 'TensorFlow'}")
print(f"Image Folder: {IMAGE_FOLDER}")
print(f"Labels JSON: {LABELS_JSON}")
print(f"Output Folder: {OUTPUT_FOLDER}") # Now shows the restored default
print(f"--------------------")


# Check if the model file exists before proceeding
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please ensure the model file exists or use the --model argument correctly.")
    sys.exit(1)

# Check if data folder and label file exist
if not os.path.exists(IMAGE_FOLDER):
    print(f"Error: Image folder '{IMAGE_FOLDER}' not found!")
    sys.exit(1)
if not os.path.exists(LABELS_JSON):
    print(f"Error: Labels JSON file '{LABELS_JSON}' not found!")
    sys.exit(1)

# Load labels from JSON file
labels_data = {}
try:
    with open(LABELS_JSON, "r") as f:
        data = json.load(f)
        # Use the folder name as the key, assuming one JSON per folder
        labels_data[os.path.basename(IMAGE_FOLDER)] = data
except FileNotFoundError:
    print(f"Error: Label file {LABELS_JSON} not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {LABELS_JSON}.")
    sys.exit(1)

# Load images and labels
def load_data(image_folder, labels_dict, img_size=(520, 240)):
    """Loads images (as YUV) and flattened labels."""
    images, label_grids_flat = [], []
    filenames_loaded = [] # Keep track of filenames for matching later if needed

    # Get the label data for the specific folder
    folder_key = os.path.basename(image_folder)
    if folder_key not in labels_dict:
        print(f"Error: No labels found for key '{folder_key}' in loaded JSON data.")
        return np.array([]), np.array([]), []

    labels = labels_dict[folder_key]

    print(f"Loading data from folder: {image_folder}")
    count = 0
    skipped = 0
    for filename_key, data in labels.items():
        # Construct expected image filename
        img_path = os.path.join(image_folder, f"{filename_key}.jpg")

        if not os.path.exists(img_path):
            # print(f"Warning: Image file not found, skipping: {img_path}")
            skipped += 1
            continue

        # Read the image (BGR format by default)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image, skipping: {img_path}")
            skipped += 1
            continue

        # Resize check (optional but good practice)
        if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
             print(f"Warning: Image {img_path} has unexpected size {img.shape[:2]}, expected {(img_size[1], img_size[0])}. Resizing.")
             img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_LINEAR)

        # --- Convert to YUV format (keeping 0-255 range) ---
        # This matches the training preprocessing step
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        images.append(img_yuv) # Append the YUV image

        # Get the label grid
        if "scores" not in data:
             print(f"Warning: 'scores' key missing for {filename_key} in labels JSON. Skipping.")
             # Remove the just added image if label is bad
             images.pop()
             skipped += 1
             continue

        label_grid = np.array(data["scores"])
        # Check label grid dimensions
        if label_grid.shape != (12, 32):
            print(f"Warning: Label grid for {filename_key} has unexpected shape {label_grid.shape}. Expected (12, 32). Skipping.")
            images.pop()
            skipped += 1
            continue

        # Flatten the label grid to match model output (12*32 = 384 elements)
        label_grid_flat = label_grid.flatten()
        label_grids_flat.append(label_grid_flat)
        filenames_loaded.append(filename_key)
        count += 1

    print(f"Successfully loaded {count} image-label pairs. Skipped {skipped}.")
    if count == 0:
        print("Error: No data loaded. Check image paths and label file content.")
        sys.exit(1)

    # Explicitly delete large intermediate objects if needed (though usually not required here)
    # del data
    # del labels

    return np.array(images), np.array(label_grids_flat), filenames_loaded

# Load the specified dataset
# X_TEST will contain YUV images (uint8)
# y_TEST will contain flattened labels (384 elements, float/int from JSON)
# --- POTENTIAL MEMORY HOG: Consider loading data differently if it's huge ---
# If X_TEST itself is taking >10-15GB, loading everything at once might be part of the problem.
# But let's assume load_data is okay for now, focusing on the video loop.
X_TEST, y_TEST, loaded_filenames = load_data(IMAGE_FOLDER, labels_data)

print(f"Loaded {len(X_TEST)} test images with shape {X_TEST.shape if len(X_TEST)>0 else 'N/A'}")
print(f"Loaded {len(y_TEST)} test labels with shape {y_TEST.shape if len(y_TEST)>0 else 'N/A'}")

# Define metric/loss functions needed ONLY for loading TF model with compile=True (if needed)
def intra_pred_variation(y_true, y_pred): # (code unchanged)
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    y_pred_norm = y_pred_reshaped / 255.0
    variance = tf.math.reduce_variance(y_pred_norm, axis=[1, 2, 3])
    return tf.reduce_mean(variance)
# ... (other dummy loss/metric functions remain unchanged) ...
def inter_pred_variation(y_true, y_pred):
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    y_pred_norm = y_pred_reshaped / 255.0
    mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
    variance = tf.reduce_mean(tf.square(y_pred_norm - mean_pred))
    return variance

def gradient_loss(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, [-1, 12, 32, 1])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 12, 32, 1])
    y_true_norm = y_true_reshaped / 255.0
    y_pred_norm = y_pred_reshaped / 255.0
    grad_true_x = y_true_norm[:, :, 1:, :] - y_true_norm[:, :, :-1, :]
    grad_pred_x = y_pred_norm[:, :, 1:, :] - y_pred_norm[:, :, :-1, :]
    grad_true_y = y_true_norm[:, 1:, :, :] - y_true_norm[:, :-1, :, :]
    grad_pred_y = y_pred_norm[:, 1:, :, :] - y_pred_norm[:, :-1, :, :]
    loss_x = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x))
    loss_y = tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))
    return loss_x + loss_y

def custom_obstacle_weighted_loss(var_penalty_weight=0.12, var_penalty_exponent=-15.0,
                                 diversity_penalty_weight=0.18, diversity_penalty_exponent=-20.0,
                                 alpha_grad=0.2):
    """Dummy loss function structure for loading model"""
    def loss_function(y_true, y_pred):
        # Simplified MSE for loading purposes if compile=True is needed
        return tf.reduce_mean(tf.square(y_true - y_pred))
    return loss_function

def scaled_sigmoid(x): # (code unchanged)
    return 255.0 * tf.nn.sigmoid(x)

# Load model based on the selected type
def load_model_eval(): # (code unchanged)
    if args.onnx:
        print(f"Loading ONNX model from {MODEL_PATH}...")
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # --- Consider forcing CPU if GPU suspected, but let's profile first ---
            # providers=['CPUExecutionProvider']
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(MODEL_PATH, session_options, providers=providers)
            print(f"ONNX session created. Available providers: {session.get_providers()}")
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
        # --- Consider disabling GPU if TF suspected, but let's profile first ---
        # try:
        #     tf.config.set_visible_devices([], 'GPU')
        #     print("--- DISABLING GPU FOR TENSORFLOW ---")
        # except Exception as e:
        #     print(f"Could not disable GPU, may already be configured: {e}")

        print(f"Loading TensorFlow model from {MODEL_PATH}...")
        try:
            custom_objects = {'scaled_sigmoid': scaled_sigmoid}
            try:
                model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                print("TF Model loaded successfully with compile=False")
            except Exception as e_compile_false:
                 print(f"Warning: Failed to load model with compile=False: {e_compile_false}")
                 print("Trying to load with compile=True (may need dummy loss/metrics)...")
                 custom_objects_compile = custom_objects.copy()
                 custom_objects_compile.update({
                     'custom_obstacle_weighted_loss': custom_obstacle_weighted_loss,
                     'gradient_loss': gradient_loss,
                     'intra_pred_variation': intra_pred_variation,
                     'inter_pred_variation': inter_pred_variation,
                     'loss_function': custom_obstacle_weighted_loss()
                 })
                 try:
                     model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects_compile, compile=True)
                     print("TF Model loaded successfully with compile=True")
                 except Exception as e_compile_true:
                     print(f"Error: Failed to load model with compile=True as well: {e_compile_true}")
                     print("Check custom objects and model saving process.")
                     sys.exit(1)
            model.summary(line_length=120)
            return model
        except Exception as e:
            print(f"Error loading TensorFlow model: {str(e)}")
            sys.exit(1)

# --- process_single_image and save_grid_to_txt remain unchanged ---
def process_single_image(image_path, model_session):
    # ... (code unchanged) ...
    print(f"Processing single image: {image_path}")
    if not os.path.exists(image_path): print(f"Error: Image file {image_path} not found"); sys.exit(1)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: print(f"Error: Could not read image {image_path}"); sys.exit(1)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    input_data = np.expand_dims(img_yuv, axis=0).astype(np.float32)
    if args.onnx:
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        pred = model_session.run([output_name], {input_name: input_data})[0]
    else:
        pred = model_session.predict(input_data)
    if pred.shape == (1, 384): pred_grid = pred[0].reshape(12, 32)
    else:
        print(f"Warning: Unexpected prediction shape {pred.shape}. Attempting reshape.")
        try: pred_grid = pred.flatten().reshape(12, 32)
        except ValueError: print("Error: Cannot reshape prediction to (12, 32)."); pred_grid = np.zeros((12, 32))
    return img_bgr, pred_grid

def save_grid_to_txt(grid, output_path):
    # ... (code unchanged) ...
    with open(output_path, 'w') as f:
        f.write("# Grid values (12x32)\n# Row,Column,Value\n")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                f.write(f"{i},{j},{grid[i, j]:.8f}\n")
        f.write("\n\n# 2D Grid Representation (rounded to 2 decimal places):\n")
        for i in range(grid.shape[0]):
            row_str = " ".join([f"{grid[i, j]:7.2f}" for j in range(grid.shape[1])])
            f.write(f"# {row_str}\n")


# Add a function to create prediction video from all images
def create_prediction_video(model_session, image_folder, labels_dict, output_folder):
    """
    Process all images in a folder, generate predictions, and create a video
    of the predictions in order of their numerical IDs. (Optimized + Debugged)
    """
    print(f"Creating prediction video for all images in {image_folder}...")

    folder_key = os.path.basename(image_folder)
    if folder_key not in labels_dict: print(f"Error: No labels found for key '{folder_key}'..."); return

    labels = labels_dict[folder_key]
    # --- Make sure we get actual labels data, not just keys ---
    # Filter out non-digit keysrobustly before sorting
    valid_ids = {}
    for k, v in labels.items():
        if k.isdigit():
            valid_ids[int(k)] = v # Store value if needed later, otherwise just k

    if not valid_ids: print("No valid numerical image IDs found."); return

    all_sorted_ids = sorted(valid_ids.keys())
    print(f"Found {len(all_sorted_ids)} image IDs to process")

    # --- Determine Frame Size (Code seems okay, but ensure cleanup) ---
    first_id = str(all_sorted_ids[0])
    first_img_path = os.path.join(image_folder, f"{first_id}.jpg")
    if not os.path.exists(first_img_path): print(f"Error: First image {first_img_path} not found."); return
    img_bgr = cv2.imread(first_img_path)
    if img_bgr is None: print(f"Error: Could not read image {first_img_path}"); return

    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    input_data = np.expand_dims(img_yuv.astype(np.float32), axis=0)

    # --- Variables for inference result ---
    pred = None
    pred_grid = None

    if args.onnx:
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        pred = model_session.run([output_name], {input_name: input_data})[0]
    else:
        # TF inference might allocate GPU memory, keep an eye on this
        pred = model_session.predict(input_data)

    if pred is not None and pred.shape == (1, 384):
         pred_grid = pred[0].reshape(12, 32)
    elif pred is not None: # Handle potential other shapes if possible
         try: pred_grid = pred.flatten().reshape(12,32)
         except: pred_grid = np.zeros((12,32)) # Fallback
         print(f"Warning: Reshaped pred from {pred.shape} for first image.")
    else: # If inference failed somehow
         pred_grid = np.zeros((12,32))
         print("Warning: Inference failed for first image, using zero grid.")


    # Create the figure ONCE to get dimensions, then close it
    fig_dim = plt.figure(figsize=(12, 8)) # Use different name to avoid conflicts
    plt.imshow(pred_grid, cmap='viridis', vmin=0, vmax=255)
    plt.title(f"ID: {first_id}") # Use dummy ID ok here
    plt.colorbar(label='Safety Score (0-255)')
    plt.tight_layout()

    buf_dim = io.BytesIO()
    fig_dim.savefig(buf_dim, format='png', dpi=fig_dim.dpi)
    buf_dim.seek(0)
    temp_img = cv2.imdecode(np.frombuffer(buf_dim.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    frame_height, frame_width, _ = temp_img.shape
    # --- Explicit Cleanup ---
    plt.close(fig_dim)
    buf_dim.close()
    del pred, pred_grid, input_data, img_yuv, img_bgr, temp_img, fig_dim, buf_dim
    #gc.collect() # Force GC after setup
    # --- End Explicit Cleanup ---

    print(f"Video frame dimensions: {frame_width}x{frame_height}")

    # Set up VideoWriter
    fps = 15
    video_path = os.path.join(output_folder, f"{folder_key}_predictions.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened(): print(f"Error: Could not create video writer for {video_path}"); return

    # Process all images in order
    processed_count = 0
    for id_num in all_sorted_ids:
        id_str = str(id_num)
        img_path = os.path.join(image_folder, f"{id_str}.jpg")

        if not os.path.exists(img_path): print(f"Warning: Image {img_path} not found, skipping."); continue

        # --- Minimal variables inside loop ---
        img_bgr_loop = cv2.imread(img_path)
        if img_bgr_loop is None: print(f"Warning: Failed to read image {img_path}, skipping."); continue

        img_yuv_loop = cv2.cvtColor(img_bgr_loop, cv2.COLOR_BGR2YUV)
        input_data_loop = np.expand_dims(img_yuv_loop.astype(np.float32), axis=0)
        del img_bgr_loop, img_yuv_loop # Delete intermediate images early

        # --- Inference ---
        pred_loop = None
        pred_grid_loop = None
        try:
            if args.onnx:
                # Reuse names if possible, otherwise get them again (safer)
                input_name = model_session.get_inputs()[0].name
                output_name = model_session.get_outputs()[0].name
                pred_loop = model_session.run([output_name], {input_name: input_data_loop})[0]
            else:
                pred_loop = model_session.predict(input_data_loop)

            if pred_loop is not None and pred_loop.shape == (1, 384):
                 pred_grid_loop = pred_loop[0].reshape(12, 32)
            elif pred_loop is not None : # Handle potential other shapes
                 try: pred_grid_loop = pred_loop.flatten().reshape(12,32)
                 except: pred_grid_loop = np.zeros((12,32))
            else: # Fallback
                 pred_grid_loop = np.zeros((12,32))

        except Exception as e:
            print(f"Error during inference for ID {id_str}: {e}")
            pred_grid_loop = np.zeros((12,32)) # Use dummy grid on error

        del input_data_loop, pred_loop # Delete inference data early

        # --- Visualization ---
        fig_loop = plt.figure(figsize=(12, 8))
        plt.imshow(pred_grid_loop, cmap='viridis', vmin=0, vmax=255)
        plt.title(f"ID: {id_str}")
        plt.colorbar(label='Safety Score (0-255)')
        plt.tight_layout()

        buf_loop = io.BytesIO()
        try:
            fig_loop.savefig(buf_loop, format='png', dpi=fig_loop.dpi)
            buf_loop.seek(0)
            frame = cv2.imdecode(np.frombuffer(buf_loop.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error during plot saving/decoding for ID {id_str}: {e}")
            frame = None # Ensure frame is None if error occurs
        finally:
            # --- Explicit Cleanup (inside loop) ---
            plt.close(fig_loop)
            buf_loop.close()
            del pred_grid_loop, fig_loop, buf_loop # Delete plot objects

        # --- Write Frame ---
        if frame is not None:
             video_writer.write(frame)
             del frame # Delete frame after writing
        else:
             print(f"Warning: Failed to generate/write frame for ID {id_str}, skipping.")

        # --- Force Garbage Collection Periodically ---
        processed_count += 1
        if processed_count % 100 == 0: # Adjust frequency as needed
            print(f"Processed {processed_count}/{len(all_sorted_ids)} images...")
            gc.collect() # Force garbage collection

    # --- Final Cleanup ---
    video_writer.release()
    del labels, valid_ids, all_sorted_ids # Clean up label data

    print(f"Video creation completed! {processed_count} frames processed.")
    print(f"Video saved to: {video_path}")


# Load the appropriate model
model_eval = load_model_eval()

# --- Main execution logic (single image, video, random samples) ---
if args.single:
    # ... (single image code unchanged) ...
    img_bgr, pred_grid = process_single_image(args.single, model_eval)
    base_name = os.path.splitext(os.path.basename(args.single))[0]
    output_folder_single = OUTPUT_FOLDER
    os.makedirs(output_folder_single, exist_ok=True)
    txt_output = os.path.join(output_folder_single, f"{base_name}_grid.txt")
    save_grid_to_txt(pred_grid, txt_output)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Input Image (RGB)"); plt.axis('off')
    plt.subplot(1, 2, 2); im = plt.imshow(pred_grid, cmap='viridis', vmin=0, vmax=255); plt.title("Predicted Safety Grid"); plt.colorbar(im, label='Safety Score (0-255)'); plt.axis('off')
    plt.suptitle(f"Safety Grid Prediction for {base_name}"); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    viz_output = os.path.join(output_folder_single, f"{base_name}_viz.png")
    plt.savefig(viz_output); print("Showing visualization..."); plt.show(); plt.close() # Close figure after show/save
    print(f"Evaluation completed for single image:\n- Grid values saved to: {txt_output}\n- Visualization saved to: {viz_output}")
    sys.exit(0)

elif args.video:
    print("Video mode enabled: processing all images and creating prediction video...")
    # Check if labels_data actually loaded correctly
    if not labels_data:
         print("Error: labels_data is empty. Cannot proceed with video generation.")
         sys.exit(1)
    # Call the decorated function
    create_prediction_video(model_eval, IMAGE_FOLDER, labels_data, OUTPUT_FOLDER)
    print(f"Prediction video creation run finished.") # Changed wording slightly
    sys.exit(0)

else: # Random sample evaluation
    # ... (random sample code mostly unchanged, ensure figures are closed) ...
    print("Running random sample evaluation...")
    num_samples = min(10, len(X_TEST))
    if num_samples == 0: print("No test samples loaded..."); sys.exit(1)
    sample_indices = random.sample(range(len(X_TEST)), num_samples)
    random_samples_yuv = X_TEST[sample_indices]
    random_labels_flat = y_TEST[sample_indices]
    inference_samples = random_samples_yuv.astype(np.float32)
    print(f"Prepared {num_samples} samples for inference...")

    # --- Nested function `run_inference` - make sure errors are handled ---
    def run_inference(model_session_inner, samples):
        if args.onnx:
            # ... (ONNX inference unchanged) ...
            try:
                input_name = model_session_inner.get_inputs()[0].name; output_name = model_session_inner.get_outputs()[0].name
                predictions_list = []
                for sample in samples:
                    input_data = np.expand_dims(sample, axis=0)
                    pred = model_session_inner.run([output_name], {input_name: input_data})[0]
                    predictions_list.append(pred[0])
                return np.array(predictions_list)
            except Exception as e: print(f"Error during ONNX inference: {str(e)}"); sys.exit(1)
        else:
            # ... (TF inference unchanged) ...
            try:
                 predictions_tf = model_session_inner.predict(samples)
                 return predictions_tf
            except Exception as e: print(f"Error during TF inference: {str(e)}"); sys.exit(1)

    predictions = run_inference(model_eval, inference_samples)

    # --- Metrics Calculation (unchanged) ---
    print("Calculating evaluation metrics...")
    y_true_flat = random_labels_flat.astype(np.float32)
    y_pred_flat = predictions.astype(np.float32)
    y_pred_flat = np.clip(y_pred_flat, 0, 255)
    y_true_grid = y_true_flat.reshape(-1, 12, 32)
    y_pred_grid = y_pred_flat.reshape(-1, 12, 32)
    mae = np.mean(np.abs(y_true_grid - y_pred_grid))
    mse = np.mean(np.square(y_true_grid - y_pred_grid))
    # ... (print metrics etc) ...
    print(f"\nEvaluation Results for {'ONNX' if args.onnx else 'TensorFlow'} model:")
    print(f"MAE: {mae:.4f} (0-255)"); print(f"MSE: {mse:.4f}"); print(f"RMSE: {np.sqrt(mse):.4f} (0-255)")
    y_true_norm = y_true_grid / 255.0; y_pred_norm = y_pred_grid / 255.0
    mae_norm = np.mean(np.abs(y_true_norm - y_pred_norm)); mse_norm = np.mean(np.square(y_true_norm - y_pred_norm))
    print(f"Normalized MAE: {mae_norm:.4f}"); print(f"Normalized MSE: {mse_norm:.4f}")

    # --- Visualization (ensure plt.close()) ---
    print("Visualizing results...")
    for i in range(num_samples):
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1); img_rgb_display = cv2.cvtColor(random_samples_yuv[i], cv2.COLOR_YUV2RGB); plt.imshow(img_rgb_display); plt.title("Input Image (RGB)"); plt.axis('off')
        truth_grid = y_true_grid[i]; plt.subplot(1, 3, 2); im_true = plt.imshow(truth_grid, cmap='viridis', vmin=0, vmax=255); plt.title("Ground Truth"); plt.colorbar(im_true, fraction=0.046, pad=0.04); plt.axis('off')
        pred_grid = y_pred_grid[i]; plt.subplot(1, 3, 3); im_pred = plt.imshow(pred_grid, cmap='viridis', vmin=0, vmax=255); plt.title("Prediction"); plt.colorbar(im_pred, fraction=0.046, pad=0.04); plt.axis('off')
        plt.suptitle(f"Sample {i+1} (Index {sample_indices[i]}) - MAE: {np.mean(np.abs(truth_grid - pred_grid)):.2f}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"sample_{i+1}_comparison.png"))
        plt.close() # Ensure figure is closed

    print(f"Evaluation of {'ONNX' if args.onnx else 'TensorFlow'} model completed successfully!")
    print(f"Visualization images saved to {OUTPUT_FOLDER} directory")

    # --- Save Metrics (unchanged) ---
    metrics_file_path = os.path.join(OUTPUT_FOLDER, "evaluation_metrics.txt")
    with open(metrics_file_path, "w") as f:
         # ... (writing metrics unchanged) ...
        f.write(f"--- Evaluation Summary ---\nModel Type: {'ONNX' if args.onnx else 'TensorFlow'}\nModel Path: {MODEL_PATH}\nDataset Folder: {IMAGE_FOLDER}\nLabels JSON: {LABELS_JSON}\nNum Samples: {num_samples}\n")
        f.write(f"---------------------------\nMAE (0-255): {mae:.4f}\nMSE (0-255^2): {mse:.4f}\nRMSE (0-255): {np.sqrt(mse):.4f}\n")
        f.write(f"Norm MAE (0-1): {mae_norm:.4f}\nNorm MSE (0-1): {mse_norm:.4f}\n---------------------------\n")
        f.write(f"Pred Range: {y_pred_grid.min():.4f} - {y_pred_grid.max():.4f}\nGT Range: {y_true_grid.min():.4f} - {y_true_grid.max():.4f}\n")

    print(f"Evaluation metrics saved to {metrics_file_path}")

    # --- Final Cleanup for random sample mode ---
    del X_TEST, y_TEST, loaded_filenames # Explicitly delete potentially large loaded data
    del random_samples_yuv, random_labels_flat, inference_samples, predictions
    del y_true_flat, y_pred_flat, y_true_grid, y_pred_grid
