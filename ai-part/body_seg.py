import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import time
from typing import List

# Body segmentation model setup
from common import create_preprocessor, TaskType
from test_segmentation import SapiensSegmentation

# Classes for segmentation
classes = [
    "Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand",
    "Left Lower Arm", "Left Lower Leg", "Left Shoe", "Left Sock", "Left Upper Arm",
    "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand", "Right Lower Arm",
    "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
    "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth", "Tongue"
]

# Random color palette for segmentation visualization
random = np.random.RandomState(11)
colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]

# Function to draw segmentation map
def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img

# Convert segmentation to Unreal Engine format
def convert_to_unreal_engine_format(segmentation_map, image_width, image_height):
    body_parts = {}

    for part_id, part_name in enumerate(classes):
        coords = np.argwhere(segmentation_map == part_id)
        if coords.size > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin

            # Normalize the sizes
            normalized_width = width / image_width
            normalized_height = height / image_height

            body_parts[part_name] = {
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "normalized_width": round(normalized_width, 4),
                "normalized_height": round(normalized_height, 4)
            }

    return body_parts

# Compute detailed body metrics
def compute_detailed_body_metrics(segmentation_data, image_width, image_height):
    """
    Compute detailed body metrics and ensure all predefined keys are included.
    """
    body_metrics = {
        "age": {
            "set": 70,
            "age_color": 1.0,
            "age_wrinkles": 6.0
        },
        "keys": {
            "Forearm Length": 0,
            "Forearm Thickness": 0,
            "Hand Length": 0,
            "Hand Thickness": 0,
            "Hand Width": 0,
            "Upper Arm Length": 0,
            "Upper Arm Thickness": 0,
            "Neck Length": 0,
            "Neck Thickness": 0,
            "Foot Length": 0,
            "Shin Length": 0,
            "Shin Thickness": 0,
            "Thigh Length": 0,
            "Thigh Thickness": 0,
            "height_150": round(image_height / 150, 4),
            "height_200": round(image_height / 200, 4),
            "muscular": 0,
            "overweight": 0,
            "skinny": 0.42,
            "Back Muscles": 0,
            "Biceps": 0,
            "Calves Muscles": 0,
            "Chest Muscles": 0,
            "Forearm Muscles": 0,
            "Hamstring Muscles": 0,
            "Lower Butt Muscles": 0,
            "Quad Muscles": 0,
            "Shoulder Muscles": 0,
            "Traps Muscles": 0,
            "Triceps": 0,
            "Upper Butt Muscles": 0,
            "Stylized": 0,
            "Belly Size": 0,
            "Breast Size": 0,
            "Chest Height": 0,
            "Chest Width": 0,
            "Hips Height": 0,
            "Hips Size": 0,
            "Shoulder Width": 0,
            "Waist Thickness": 0,
            "asian": -0.36,
            "black": 0.0,
            "caucasian": 0.0,
            "variation_1": 1.0,
            "variation_10": 0.0,
            "variation_11": 0.0,
            "variation_2": 0.0,
            "variation_3": 0.0,
            "variation_4": 0.0,
            "variation_5": 0.0,
            "variation_6": 0.0,
            "variation_7": 0.0,
            "variation_8": 0.0,
            "variation_9": 0.0
        }
    }

    # Update specific keys based on segmentation data
    for part, data in segmentation_data.items():
        if "normalized_width" in data and "normalized_height" in data:
            if part == "Left Hand" or part == "Right Hand":
                body_metrics["keys"]["Hand Width"] = data["normalized_width"]
                body_metrics["keys"]["Hand Length"] = data["normalized_height"]

            if part == "Left Lower Arm" or part == "Right Lower Arm":
                body_metrics["keys"]["Forearm Length"] = data["normalized_height"]
                body_metrics["keys"]["Forearm Thickness"] = data["normalized_width"]

            if part == "Left Upper Arm" or part == "Right Upper Arm":
                body_metrics["keys"]["Upper Arm Length"] = data["normalized_height"]
                body_metrics["keys"]["Upper Arm Thickness"] = data["normalized_width"]

            if part == "Face Neck":
                body_metrics["keys"]["Neck Length"] = data["normalized_height"]
                body_metrics["keys"]["Neck Thickness"] = data["normalized_width"]

            if part == "Left Foot" or part == "Right Foot":
                body_metrics["keys"]["Foot Length"] = data["normalized_height"]

            if part == "Left Lower Leg" or part == "Right Lower Leg":
                body_metrics["keys"]["Shin Length"] = data["normalized_height"]
                body_metrics["keys"]["Shin Thickness"] = data["normalized_width"]

            if part == "Left Upper Leg" or part == "Right Upper Leg":
                body_metrics["keys"]["Thigh Length"] = data["normalized_height"]
                body_metrics["keys"]["Thigh Thickness"] = data["normalized_width"]

    return body_metrics

# Capture photo from webcam
def capture_photo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    print("Press 's' to take a photo. Press 'q' to quit.")

    photo = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.imshow("Adjust Position and Press 's'", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Take a photo
            photo = frame.copy()
            print("Photo captured.")
            break
        elif key == ord('q'):  # Exit without taking a photo
            break

    cap.release()
    cv2.destroyAllWindows()
    return photo

# Perform body segmentation
def perform_body_segmentation(photo):
    print("Loading body segmentation model...")
    model = SapiensSegmentation()

    print("Performing body segmentation...")
    start_time = time.time()
    segmentation_map = model(photo)
    end_time = time.time()

    print(f"Segmentation completed in {end_time - start_time:.2f} seconds.")
    segmentation_image = draw_segmentation_map(segmentation_map)

    # Display segmentation result
    cv2.imshow("Body Segmentation", segmentation_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmentation_map

# Main function
if __name__ == "__main__":
    print("Starting the application...")

    # Step 1: Open camera and capture photo
    photo = capture_photo()
    if photo is None:
        print("No photo captured. Exiting...")
        exit()

    # Step 2: Perform body segmentation
    segmentation_map = perform_body_segmentation(photo)

    # Step 3: Convert segmentation to Unreal Engine format
    image_height, image_width = photo.shape[:2]
    segmentation_data = convert_to_unreal_engine_format(segmentation_map, image_width, image_height)

    # Step 4: Compute detailed body metrics
    detailed_metrics = compute_detailed_body_metrics(segmentation_data, image_width, image_height)

    # Save results to JSON
    output_file = "detailed_body_metrics.json"
    with open(output_file, "w") as f:
        json.dump(detailed_metrics, f, indent=4)

    print(f"Detailed body metrics saved to {output_file}.")
