from flask import Flask, jsonify, request, render_template
import os
import cv2
from test_segmentation import SapiensSegmentation
from body_seg import convert_to_unreal_engine_format, compute_detailed_body_metrics
from deepface import DeepFace
import stone
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = SapiensSegmentation()
logging.info("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('capture.html')

@app.route('/upload/<image_type>', methods=['POST'])
def upload(image_type):
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"{image_type}_image.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    logging.info(f"{image_type.capitalize()} image saved to {save_path}")
    return jsonify({'path': save_path}), 200

def analyze_face(image_path):
    try:
        # DeepFace Analysis
        analysis = DeepFace.analyze(image_path, actions=['age', 'gender'])
        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']

        # Skin Tone Analysis
        stone_result = stone.process(image_path, image_type="color", return_report_image=False)
        dominant_color = stone_result.get("faces", [{}])[0].get("skin_tone", "#FFFFFF")
        
        skin_tone_value = {
            "#9D7A54": 1.0,  
            "#C2967B": 2.0,  
            "#715541": 3.0   
        }.get(dominant_color.upper(), 1.5)  

        redness = stone_result.get("faces", [{}])[0].get("redness", 0.5)
        saturation = stone_result.get("faces", [{}])[0].get("saturation", 0.5)

        return {
            "age": age,
            "gender": gender,
            "skin_tone": skin_tone_value,
            "redness": round(redness, 4),
            "saturation": round(saturation, 4)
        }
    except Exception as e:
        logging.error(f"Error during face analysis: {e}")
        return None

def capture_image(image_type):
    saved_image_path = f'./uploads/{image_type}_image.jpg'
    image = cv2.imread(saved_image_path)
    if image is None:
        raise RuntimeError(f"Failed to load {image_type} image from {saved_image_path}.")
    return image

def create_complete_output(template, actual_output):
  
    for key, value in template.items():
        if key not in actual_output:
            actual_output[key] = value
        elif isinstance(value, dict):
            create_complete_output(value, actual_output[key])
    return actual_output

@app.route('/process', methods=['GET'])
def process_images():
    try:
        # Define the template JSON structure
        template_output = {
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
                "height_150": 0.0,
                "height_200": 0.0,
                "muscular": 0,
                "overweight": 0,
                "skinny": 0.0,
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
                "asian": 0.0,
                "black": 0.0,
                "caucasian": 0.0
            },
            "skin": {
                "tone": 0.0,
                "redness": 0.0,
                "saturation": 0.0,
                "normal_strength": 4.0,
                "roughness_multiplier": 1.5,
                "freckles": 0.0,
                "splotches": 0.0,
                "texture.set": "textures\\male\\Default 4K\\Male 08.png",
                "cavity_strength": 1.3333333730697632,
                "gender_specific": {
                    "mustache_shadow": 0.0,
                    "beard_shadow": 0.0
                }
            },
            "eyes": {
        "pupil_color": [
            0.1441284716129303,
            0.06847816705703735,
            0.06301001459360123,
            1.0
        ],
        "sclera_color": [
            1.0,
            1.0,
            1.0,
            1.0
        ]
    },
    "height": {
        "set": 170
    },
    "hair": {
        "eyebrows": {
            "set": "Eyebrows_002",
            "lightness": 1.0632911920547485,
            "redness": 0.7721518874168396,
            "roughness": 0.44999998807907104,
            "salt_and_pepper": 0.0,
            "roots": 0.0,
            "root_lightness": 0.5,
            "root_redness": 0.0,
            "roots_hue": 0.5,
            "fast_or_accurate": 0.0,
            "hue": 0.5
        },
        "regular_hair": {
            "set": "hair\\head\\male\\Aged\\Bald Top Combover.json",
            "lightness": 0.9620253443717957,
            "redness": 0.32278478145599365,
            "roughness": 0.44999998807907104,
            "salt_and_pepper": 0.297468364238739,
            "roots": 0.0,
            "root_lightness": 0.0,
            "root_redness": 0.8999999761581421,
            "roots_hue": 0.5,
            "fast_or_accurate": 0.0,
            "hue": 0.5
        },
        "face_hair": {
            "set": "hair/face_hair/Other/Stubble_Long.json",
            "lightness": 0.0,
            "redness": 1.0,
            "roughness": 0.44999998807907104,
            "salt_and_pepper": 0.4430379867553711,
            "roots": 0.0,
            "root_lightness": 0.0,
            "root_redness": 0.8999999761581421,
            "roots_hue": 0.5,
            "fast_or_accurate": 0.0,
            "hue": 0.5
        }
    },
    "clothing": {
        "outfit": {
            "set":None
        },
        "footwear": {
            "set":None
        }
    }
        }

        # Face Analysis
        face_image_path = './uploads/face_image.jpg'
        face_results = analyze_face(face_image_path)

        if not face_results:
            return jsonify({'error': 'Face analysis failed'}), 500

        logging.debug(f"Face Analysis Results: {face_results}")

        # Body Segmentation
        body_image = capture_image("body")
        image_height, image_width = body_image.shape[:2]
        logging.info("Running the model on the body image...")
        segmentation_map = model(body_image)
        segmentation_data = convert_to_unreal_engine_format(segmentation_map, image_width, image_height)
        body_metrics = compute_detailed_body_metrics(segmentation_data, image_width, image_height)

        # Ensure "skin" key exists in body_metrics
        if "skin" not in body_metrics:
            body_metrics["skin"] = {}

        # Update body metrics with face results
        body_metrics["age"]["set"] = face_results["age"]
        body_metrics["skin"]["tone"] = face_results["skin_tone"]
        body_metrics["skin"]["redness"] = face_results["redness"]
        body_metrics["skin"]["saturation"] = face_results["saturation"]

        # Complete the JSON structure
        complete_output = create_complete_output(template_output, body_metrics)

        logging.debug(f"Final JSON Output: {complete_output}")

        # Return the combined result as JSON
        return jsonify(complete_output), 200

    except RuntimeError as e:
        logging.error(f"Image Taken")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Image Taken .")
        return jsonify({"Image taken"}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
