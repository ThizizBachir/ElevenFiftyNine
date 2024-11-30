import stone
import cv2
from json import dumps

def get_skin_tone_and_json(image_path, face_id=1):
    try:
        result = stone.process(image_path, image_type="color", return_report_image=True)
        
        report_images = result.pop("report_images")
        
        result_json = dumps(result, indent=4)  
        
        skin_tone = result.get("color", {}).get("name", "Unknown")
        
        if report_images and face_id < len(report_images):
            cv2.imshow("Result", report_images[face_id])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_json, skin_tone
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, "Error processing image"

image_path = "./pdp.jpg"

result_json, skin_tone = get_skin_tone_and_json(image_path)

print("Result JSON:", result_json)
print("Detected Skin Tone:", skin_tone)
