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
outfit_map = {
    1: 'outfits\\male\\Casual\\Casual_Weekday.blend',
    2: 'outfits\\male\\Casual\\Relaxed_Office.blend',
    3: 'outfits\\male\\Casual\\Skinny_Look.blend',
    4: 'outfits\\male\\Casual\\Smart_Casual.blend',
    5: 'outfits\\male\\Casual\\Stylish_Casual.blend',
    6: 'outfits\\male\\Casual\\Weekend_Warrior.blend',
    7: 'outfits\\male\\Extra Outfits Pack\\Bomber look.blend',
    8: 'outfits\\male\\Extra Outfits Pack\\Flight Suit.blend',
    9: 'outfits\\male\\Extra Outfits Pack\\Golf Day.blend',
    10: 'outfits\\male\\Extra Outfits Pack\\Lab Tech.blend',
    11: 'outfits\\male\\Extra Outfits Pack\\On the road.blend',
    12: 'outfits\\male\\Extra Outfits Pack\\Pirate.blend',
    13: 'outfits\\male\\Extra Outfits Pack\\Relaxed Fit.blend',
    14: 'outfits\\male\\Office\\New_Intern.blend',
    15: 'outfits\\male\\Office\\Open_Suit.blend',
    16: 'outfits\\male\\Office\\Relaxed_Dresscode.blend',
    17: 'outfits\\male\\Office\\Stock_Exchange.blend',
    18: 'outfits\\male\\Office\\Suit_N_Tie.blend',
    19: 'outfits\\male\\Office\\Summer_Lawyer.blend',
    20: 'outfits\\male\\Summer\\BBQ_Barry.blend',
    21: 'outfits\\male\\Summer\\Beach_Day.blend',
    22: 'outfits\\male\\Summer\\Office_Excursion.blend',
    23: 'outfits\\male\\Winter\\Frosty_Evening.blend'
}

footwear_map = {
    1: 'footwear\\male\\Extra Outfits Pack\\Pirate Boots.blend',
    2: 'footwear\\male\\Office\\HG_Leather_Dress_Shoe.blend',
    3: 'footwear\\male\\Office\\HG_Suede_Dress_Shoe.blend',
    4: 'footwear\\male\\Outdoor\\HG_Garden_Boot_Male.blend',
    5: 'footwear\\male\\Sneakers\\HG_Leather_Sneakers_Male.blend',
    6: 'footwear\\male\\Sneakers\\HG_Suede_Sneakers_Male.blend',
    7: 'footwear\\male\\Sports\\HG_Running_Shoe_1.blend',
    8: 'footwear\\male\\Sports\\HG_Running_Shoe_2.blend'
}

# Path to your JSON file
file_path = r"E:\Download\detailed_body_metrics (2).json"

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)


outfit=data['clothing']['outfit']['set']
footwear=data['clothing']['footwear']['set']



# Loop through all elements in the JSON, but only process "keys"

                            
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
            
                     
                
                
                
                
                
                               


process_json_Face_and_Body(data)
process_json_Skin(data)
process_json_Height(data)
my_existing_human.clothing.footwear.set(footwear_map[footwear])
my_existing_human.clothing.outfit.set(outfit_map[outfit])




