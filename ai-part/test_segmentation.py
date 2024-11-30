import time
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import create_preprocessor

model_path='./model/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2'
class SapiensSegmentationType(Enum):
    SEGMENTATION_03B = "sapiens-seg-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
    SEGMENTATION_06B = "sapiens-seg-0.6b-torchscript/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
    SEGMENTATION_1B = "sapiens-seg-1b-torchscript/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"


random = np.random.RandomState(11)
classes = ["Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand", "Left Lower Arm", "Left Lower Leg",
           "Left Shoe", "Left Sock", "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand",
           "Right Lower Arm", "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
           "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth", "Tongue"]

colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  
colors = colors[:, ::-1]


def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img


def postprocess_segmentation(results: torch.Tensor, img_shape: Tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    segmentation_map = logits.argmax(dim=0, keepdim=True)

    segmentation_map = segmentation_map.float().numpy().squeeze()

    return segmentation_map


class SapiensSegmentation():
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_03B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        path = model_path
        model = torch.jit.load(path)
        model = model.eval()
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])

        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map


if __name__ == "__main__":
    type = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "./pdp.jpg"
    img = cv2.imread(img_path)

    model_type = SapiensSegmentationType.SEGMENTATION_03B
    estimator = SapiensSegmentation(model_type)
    if img is not None:

        start = time.perf_counter()
        segmentations = estimator(img)
        print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

        segmentation_img = draw_segmentation_map(segmentations)
        combined = cv2.addWeighted(img, 0.5, segmentation_img, 0.5, 0)

        cv2.imshow("segmentation_map", combined)
        cv2.waitKey(0)
    print("Starting real-time segmentation...")
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            segmentations = estimator(frame)
            segmentation_img = draw_segmentation_map(segmentations)

            combined = cv2.addWeighted(frame, 0.5, segmentation_img, 0.5, 0)

            cv2.imshow("Real-Time Segmentation", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user. Closing...")
    finally:
        cap.release()
        cv2.destroyAllWindows()