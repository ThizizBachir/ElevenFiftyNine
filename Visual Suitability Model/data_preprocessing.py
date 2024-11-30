import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image



def preprocess_image(img_array, target_size=(299, 299)):
    """
    Preprocess image to ensure it's compatible with InceptionV3 input

    Args:
    img_array (numpy.ndarray or PIL.Image): Input image
    target_size (tuple): Target image size

    Returns:
    torch.Tensor: Preprocessed image tensor
    """

    # If it's a PIL Image, convert to numpy array first
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)

    # Handle different input array shapes
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.ndim == 3:
        # Handle various channel configurations
        if img_array.shape[0] in [1, 3] and img_array.shape[0] != img_array.shape[-1]:
            # Channel-first image
            img_array = np.transpose(img_array, (1, 2, 0))

        # Ensure 3 channels
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.shape[-1] != 3:
            # If not 3 channels, convert to grayscale and repeat
            if img_array.shape[-1] > 3:
                img_array = img_array[..., :3]
            else:
                # Fallback for unexpected shapes
                img_array = np.mean(img_array, axis=-1)
                img_array = np.stack([img_array, img_array, img_array], axis=-1)

    # Ensure uint8 type
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_array)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Transform image and add batch dimension
    img_tensor = transform(pil_img).unsqueeze(0)

    return img_tensor


def extract_image_features(img_path, target_size=(299, 299)):
    """
    Robust feature extraction from InceptionV3
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_array = Image.open(img_path)
    # Preprocess image
    img_tensor = preprocess_image(img_array, target_size)

    # Move input tensor to the same device as the model
    img_tensor = img_tensor.to(device)

    # Load pre-trained InceptionV3 model
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.eval()

    # Move model to GPU
    model = model.to(device)

    # Custom feature extractor
    class InceptionFeatureExtractor(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            # Manually select layers based on the current Inception3 implementation
            self.features = torch.nn.Sequential(
                original_model.Conv2d_1a_3x3,  # Corrected layer name
                original_model.Conv2d_2a_3x3,  # Corrected layer name
                original_model.Conv2d_2b_3x3,  # Corrected layer name
                original_model.maxpool1,
                original_model.Conv2d_3b_1x1,
                original_model.Conv2d_4a_3x3,
                original_model.maxpool2,
                original_model.Mixed_5b,
                original_model.Mixed_5c,
                original_model.Mixed_5d,
                original_model.Mixed_6a,
                original_model.Mixed_6b,
                original_model.Mixed_6c,
                original_model.Mixed_6d,
                original_model.Mixed_6e,
                original_model.Mixed_7a,
                original_model.Mixed_7b,
                original_model.Mixed_7c
            )

            # Global average pooling and feature reduction
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_reducer = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU()
            )

        def forward(self, x):
            # Extract features through all layers
            x = self.features(x)

            # Global average pooling
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)

            # Reduce features
            x = self.feature_reducer(x)

            return x

    # Create feature extractor
    feature_extractor = InceptionFeatureExtractor(model)

    # Move feature extractor to GPU
    feature_extractor = feature_extractor.to(device)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # Move features back to CPU and convert to numpy
    feature_vector = features.squeeze().cpu().numpy()

    return feature_vector

