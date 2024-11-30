import json
from HumGen3D import Human
import bpy

Body_attributes = [
"Forearm Length",
"Forearm Thickness",
"Hand Length",
"Hand Thickness",
"Hand Width",
"Upper Arm Length",
"Upper Arm Thickness",
"Neck Length",
"Neck Thickness",
"Foot Length",
"Shin Length",
"Shin Thickness",
"Thigh Length",
"Thigh Thickness",
"muscular",
"overweight",
"skinny",
"Back Muscles",
"Biceps",
"Calves Muscles",
"Chest Muscles",
"Forearm Muscles",
"Hamstring Muscles",
"Lower Butt Muscles",
"Quad Muscles",
"Shoulder Muscles",
"Traps Muscles",
"Triceps",
"Upper Butt Muscles",
"Stylized",
"Belly Size",
"Breast Size",
"Chest Height",
"Chest Width",
"Hips Height",
"Hips Size",
"Shoulder Width",
"Waist Thickness"
]


Age_attributes=[
    "Main Lightness",
    "Eye Hair Lightness",
    "Main Redness",
    "Eye Hair Redness",
    "Main Salt and Pepper",
    "Aged Male",
    "Aged Young",
    "Wrinkles",
    "Age Color",
    "Cavity Strength",
    "Normal Strength"
]
Face_attributes=[   
"cheek_fullness",
"cheek_zygomatic_bone",
"cheek_zygomatic_proc",
"chin_dimple",
"chin_height",
"chin_size",
"chin_width",
"ear_antihelix_shape",
"ear_height",
"ear_lobe_size",
"ear_turn",
"ear_width",
"EyeDepth",
"EyeDistance",
"EyeHeight",
"eyelid_fat_pad",
"eyelid_rotation",
"eyelid_shift_horizontal",
"eyelid_shift_vertical",
"eye_height",
"eye_orbit_size",
"eye_tilt",
"eye_width",
"jaw_location_horizontal",
"jaw_location_vertical",
"jaw_width",
"muzzle_location_horizontal",
"muzzle_location_vertical",
"lip_cupid_bow",
"lip_height",
"lip_location",
"lip_offset",
"lip_width",
"nose_angle",
"nose_bridge_height",
"nose_bridge_width",
"nose_height",
"nose_location",
"nose_nostril_flare",
"nose_nostril_turn",
"nose_tip_angle",
"nose_tip_length",
"nose_tip_size",
"nose_tip_width",
"EyeScale",
"browridge_center_size",
"browridge_loc_horizontal",
"browridge_loc_vertical",
"forehead_size",
"temple_size"]

Skin_attribute_map = {
    "tone": ("Skin_tone", 1),
    "redness": ("Skin_tone", 2),
    "saturation": ("Skin_tone", 3),
}

# Path to your JSON file
file_path = r"E:\Download\detailed_body_metrics (2).json"

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Loop through all elements in the JSON, but only process "keys"
def process_json_body(data, prefix=""):
    active_object = bpy.context.active_object
    my_existing_human = Human.from_existing(active_object)
    livekeys_list=my_existing_human.body.keys
    for key, value in data.items():
        # If the value is a dictionary and it's not "keys", continue recursion
        # Process "keys" dictionary only
        if key == "keys":
            for sub_key, sub_value in value.items():
                if sub_key in Body_attributes:
                    for item in livekeys_list:
                        if item.name == sub_key:
                            item.value =sub_value  # Change the value to the desired number

def process_json_Face(data, prefix=""):
    active_object = bpy.context.active_object
    my_existing_human = Human.from_existing(active_object)
    livekeys_list=my_existing_human.face.keys
    for key, value in data.items():
        # If the value is a dictionary and it's not "keys", continue recursion
        # Process "keys" dictionary only
        if key == "keys":
            for sub_key, sub_value in value.items():
                if sub_key in Face_attributes:
                    for item in livekeys_list:
                        if item.name == sub_key:
                            item.value =sub_value
                            
"""def process_json_Face_and_Body(data):
    active_object = bpy.context.active_object
    my_existing_human = Human.from_existing(active_object)
    livekeys_list_face=my_existing_human.face.keys
    livekeys_list_body=my_existing_human.body.keys
    for key, value in data.items():
        # If the value is a dictionary and it's not "keys", continue recursion
        # Process "keys" dictionary only
        if key == "keys":
            for sub_key, sub_value in value.items():
                if sub_key in Face_attributes:
                    for item in livekeys_list_face:
                        if item.name == sub_key:
                            item.value =sub_value
                elif sub_key in Body_attributes:
                    for item in livekeys_list_body:
                        if item.name == sub_key:
                            item.value =sub_value"""
                            
def process_json_Face_and_Body(data):
    active_object = bpy.context.active_object
    my_existing_human = Human.from_existing(active_object)
    livekeys_list_face = my_existing_human.face.keys
    livekeys_list_body = my_existing_human.body.keys

    for key, value in data.items():
        # If the value is a dictionary and it's not "keys", continue recursion
        # Process "keys" dictionary only
        if key == "keys":
            for sub_key, sub_value in value.items():
                if sub_key in Face_attributes:
                    for item in livekeys_list_face:
                        if item.name == sub_key:
                            item.value = sub_value
                            # Continue to the next sub_key after updating the value
                            break
                elif sub_key in Body_attributes:
                    for item in livekeys_list_body:
                        if item.name == sub_key:
                            item.value = sub_value
                            # Continue to the next sub_key after updating the value
                            break
                            
                            
def process_json_Skin(data):
    active_object = bpy.context.active_object
    _human = Human.from_existing(active_object) 
    skin_settings=_human.skin
    skin_nodes=skin_settings.nodes   
    for key, value in data.items():                    
        if key == "skin":
            for sub_key, sub_value in value.items():
                if sub_key in Skin_attribute_map.keys():
                        skin_nodes["Skin_tone"].inputs[Skin_attribute_map[sub_key][1]].default_value = sub_value
                        
                        
def process_json_Height(data):
    context = bpy.context
    active_object = bpy.context.active_object
    _human = Human.from_existing(active_object)
    Height_Settings=_human.height
    for key, value in data.items():                    
        if key == "height": 
            for sub_key, sub_value in value.items():
                 if sub_key =="set":
                     Height_Settings.set(sub_value, context=context, realtime=True)
            
                     
                
                
                
                
            
                               
print("++++-----++++++-------++++----++++++-----++++-----")


process_json_Face_and_Body(data)
process_json_Skin(data)




#process_json(data, prefix="")                

active_object = bpy.context.active_object


my_existing_human = Human.from_existing(active_object)

livekeys_list=my_existing_human.body.keys

for item in livekeys_list:
    print(item.name)
    if item.name == 'Neck Length':
        item.value =-1.0  # Change the value to the desired number
        break
    