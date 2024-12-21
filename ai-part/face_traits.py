import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image



MODEL_PATH = "project/model/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Download it and place it in the script's directory.")

def create_facelandmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    return mp.Image.create_from_file(image_path)
def normalize_score(score, min_val=-2, max_val=2):
    return (score - (-1)) / (1 - (-1)) * (max_val - min_val) + min_val

def face_traits(jsonFile):
    IMAGE_PATH = "project/uploads/face_image.jpg"  

    detector = create_facelandmarker()

    # Load and process the image
    image = load_image(IMAGE_PATH)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    annotated_image_path = "annotated_image.png"
    cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # print(f"Annotated image saved to '{annotated_image_path}'.")

    if detection_result.face_blendshapes:
        blendshapes = detection_result.face_blendshapes[0]
        # print("Face Blendshapes:")
        for category in blendshapes:
              normalized_score = normalize_score(category.score, -2, 2)
              # print(f"{category.category_name}: {category.score:.4f}")
              if category.category_name == "cheekPuff":
                # print("aaaaaaaaaaaaaaaaaa",jsonFile["keys"]["cheek_fullness"])
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["cheek_fullness"] = f"{number:.10f}"
              elif category.category_name == "zygomaticBone":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["cheek_zygomatic_bone"] = f"{number:.10f}"
              elif category.category_name == "chinDimple":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["chin_dimple"] = f"{number:.10f}"
              elif category.category_name == "chinHeight":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["chin_height"] = f"{number:.10f}"
              elif category.category_name == "chinSize":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["chin_size"] =f"{number:.10f}"
              elif category.category_name == "chinWidth":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["chin_width"] = f"{number:.10f}"
              elif category.category_name == "earHeight":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["ear_height"] = f"{number:.10f}"
              elif category.category_name == "earWidth":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["ear_width"] = f"{number:.10f}"
              elif category.category_name == "eyeDepth":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["eye_depth"] = f"{number:.10f}"
              elif category.category_name == "eyeDistance":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["eye_distance"] = f"{number:.10f}"
              elif category.category_name == "eyeHeight":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["eye_height"] = f"{number:.10f}"
              elif category.category_name == "noseHeight":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["nose_height"] = f"{number:.10f}"
              elif category.category_name == "noseWidth":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["nose_width"] = f"{number:.10f}"
              elif category.category_name == "lipWidth":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["lip_width"] = f"{number:.10f}"
              elif category.category_name == "lipHeight":
                number=float(f"{normalized_score:.10f}")
                jsonFile["keys"]["lip_height"] = f"{number:.10f}"
        # print(jsonFile)
        return jsonFile
    else:
        print("No face blendshapes detected.")
template_output={
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
                "height_150": 0.2941176470588235,
                "height_200": 0.0,
                "muscular": 0,
                "overweight": 0,
                "skinny": 0.41999998688697815,
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
                "asian": -0.36050158739089966,
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
                "variation_9": 0.0,
                "cheek_fullness": -0.426529193061444,
                "cheek_zygomatic_bone": -0.3397818412906088,
                "cheek_zygomatic_proc": 0.22597252464616524,
                "chin_dimple": -0.22044779616444696,
                "chin_height": 0.10308551531118769,
                "chin_size": 0.101932210262438,
                "chin_width": 0.21844671167340277,
                "ear_antihelix_shape": 0.2960981366876004,
                "ear_height": -0.2558366825846757,
                "ear_lobe_size": 0.013382304786754668,
                "ear_turn": 0.4135377462541806,
                "ear_width": 0.45049329523226167,
                "Eye Depth": -0.020148297243488586,
                "Eye Distance": -0.069108115932617,
                "Eye Height": 0.15124934472140147,
                "eyelid_fat_pad": 0.3513932596954031,
                "eyelid_rotation": -0.4951512767785743,
                "eyelid_shift_horizontal": 0.1381158698764263,
                "eyelid_shift_vertical": -0.4795948591257173,
                "eye_height": -0.13233464886747873,
                "eye_orbit_size": 0.14084363267765607,
                "eye_tilt": -1.4875490194884278,
                "eye_width": 0.5914431100299197,
                "jaw_location_horizontal": 0.6684570295972594,
                "jaw_location_vertical": 1.2843229957878532,
                "jaw_width": 0.40839215857290645,
                "muzzle_location_horizontal": -0.7639297032568753,
                "muzzle_location_vertical": 0.5933491219457957,
                "lip_cupid_bow": 0.4867029754099098,
                "lip_height": 0.2730577680513609,
                "lip_location": -0.3714075843412724,
                "lip_offset": 0.07647783716282927,
                "lip_width": -0.6465058783167801,
                "nose_angle": -0.39808056510802775,
                "nose_bridge_height": 0.005266191244486004,
                "nose_bridge_width": 0.3943328140033444,
                "nose_height": 0.5721974331021343,
                "nose_location": 0.03679741378201305,
                "nose_nostril_flare": -0.15967554956269492,
                "nose_nostril_turn": 0.42854525171273705,
                "nose_tip_angle": -0.2809719809675234,
                "nose_tip_length": 0.5360342554068008,
                "nose_tip_size": -0.8317328269066996,
                "nose_tip_width": 0.1528230548930014,
                "Eye Scale": 0.0,
                "browridge_center_size": 0.29377189446280383,
                "browridge_loc_horizontal": 0.11404012284868587,
                "browridge_loc_vertical": -0.7925576118385862,
                "forehead_size": 0.48228867651632973,
                "temple_size": 0.8081784330983255,
                "aged_male": 1.0,
                "aged_young": 1.0,
                "Male": 1.0,
                "LIVE_KEY_PERMANENT": 1.0,
                "LIVE_KEY_TEMP_skinny": 0.41999998688697815
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
face_traits(template_output)
# template_output={'age': {'set': 24, 'age_color': 1.0, 'age_wrinkles': 6.0}, 'keys': {'Forearm Length': 0, 'Forearm Thickness': 0, 'Hand Length': 0, 'Hand Thickness': 0, 'Hand Width': 0, 'Upper Arm Length': 0, 'Upper Arm Thickness': 0, 'Neck Length': 0.5167, 'Neck Thickness': 0.2875, 'Foot Length': 0, 'Shin Length': 0, 'Shin Thickness': 0, 'Thigh Length': 0, 'Thigh Thickness': 0, 'height_150': 1.6, 'height_200': 1.2, 'muscular': 0, 'overweight': 0, 'skinny': 0.42, 'Back Muscles': 0, 'Biceps': 0, 'Calves Muscles': 0, 'Chest Muscles': 0, 'Forearm Muscles': 0, 'Hamstring Muscles': 0, 'Lower Butt Muscles': 0, 'Quad Muscles': 0, 'Shoulder Muscles': 0, 'Traps Muscles': 0, 'Triceps': 0, 'Upper Butt Muscles': 0, 'Stylized': 0, 'Belly Size': 0, 'Breast Size': 0, 'Chest Height': 0, 'Chest Width': 0, 'Hips Height': 0, 'Hips Size': 0, 'Shoulder Width': 0, 'Waist Thickness': 0, 'asian': -0.36, 'black': 0.0, 'caucasian': 0.0, 'variation_1': 1.0, 'variation_10': 0.0, 'variation_11': 0.0, 'variation_2': 0.0, 'variation_3': 0.0, 'variation_4': 0.0, 'variation_5': 0.0, 'variation_6': 0.0, 'variation_7': 0.0, 'variation_8': 0.0, 'variation_9': 0.0}, 'skin': {'tone': 1.5, 'redness': 0.5, 'saturation': 0.5, 'normal_strength': 4.0, 'roughness_multiplier': 1.5, 'freckles': 0.0, 'splotches': 0.0, 'texture.set': 'textures\\male\\Default 4K\\Male 08.png', 'cavity_strength': 1.3333333730697632, 'gender_specific': {'mustache_shadow': 0.0, 'beard_shadow': 0.0}}, 'eyes': {'pupil_color': [0.1441284716129303, 0.06847816705703735, 0.06301001459360123, 1.0], 'sclera_color': [1.0, 1.0, 1.0, 1.0]}, 'height': {'set': 170}, 'hair': {'eyebrows': {'set': 'Eyebrows_002', 'lightness': 1.0632911920547485, 'redness': 0.7721518874168396, 'roughness': 0.44999998807907104, 'salt_and_pepper': 0.0, 'roots': 0.0, 'root_lightness': 0.5, 'root_redness': 0.0, 'roots_hue': 0.5, 'fast_or_accurate': 0.0, 'hue': 0.5}, 'regular_hair': {'set': 'hair\\head\\male\\Aged\\Bald Top Combover.json', 'lightness': 0.9620253443717957, 'redness': 0.32278478145599365, 'roughness': 0.44999998807907104, 'salt'hue': 0.5}, 'face_hair': {'set': 'hair/face_hair/Other/Stubble_Long.json', 'lightness': 0.0, 'redness': 1.0, 'roughness': 0.44999998807907104, 'salt_and_pepper': 0.4430379867553711, 'roots': 0.0, 'root_lightness': 0.0, 'root_redness': 0.8999999761581421, 'roots_hue': 0.5, 'fast_or_accurate': 0.0, 'hue': 0.5}}, 'clothing': {'outfit': {'set': None}, 'footwear': {'set': None}}}