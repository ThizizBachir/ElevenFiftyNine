import streamlit as st
import torch
import torchvision.transforms as transforms
from torch import  nn
from PIL import Image
import numpy as np

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


class BiLSTMOutfitModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, dropout_rate=0.7):
        """
        Outfit Compatibility Scoring Model

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): LSTM hidden layer size
            dropout_rate (float): Dropout probability
        """
        super(BiLSTMOutfitModel, self).__init__()

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,  # Multiple layers for deeper feature extraction
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Fully connected layers for compatibility scoring
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # Bidirectional, so multiply by 2
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x (tensor): Input sequence of embeddings

        Returns:
            tensor: Compatibility score
        """
        # Attention mechanism
        attn_output, _ = self.attention(x, x, x)

        # Initial hidden states
        h0 = torch.zeros(4, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(4, x.size(0), self.lstm.hidden_size).to(x.device)

        # LSTM forward pass with attention input
        out, _ = self.lstm(attn_output, (h0, c0))

        # Take the last time step
        out = out[:, -1, :]

        # Compatibility score
        return self.fc(out)

class OutfitScoringApp:
    def __init__(self, model_path):
        # Load the BiLSTM model
        self.model = self.load_model(model_path)

        # Image transformation
        self.transform = preprocess_image

    def load_model(self,model_path):
        """
        Load the Bi-LSTM model.
        This is a placeholder - replace with actual model loading logic.
        """
        model = BiLSTMOutfitModel()  # Your actual model class
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    def process_images(self, uploaded_images):
        """
        Process uploaded images and prepare them for the model
        """
        processed_images = []
        for img in uploaded_images:
            # Convert to PIL Image and transform
            pil_img = Image.open(img)
            tensor_img = self.transform(pil_img).unsqueeze(0)
            processed_images.append(tensor_img)

        # Stack images into a single tensor
        return torch.cat(processed_images, dim=0)

    def score_outfit(self, base_images, variant_images):
        """
        Score outfit variants
        """
        base_tensor = self.process_images(base_images)
        variant_tensors = [self.process_images(var_set) for var_set in variant_images]

        scores = []
        with torch.no_grad():
            # Score base outfit
            base_score = self.model(base_tensor)

            # Score variants
            for var_tensor in variant_tensors:
                var_score = self.model(var_tensor)
                scores.append(var_score.item())

        return scores


def main():
    st.title('Outfit Scoring AI Interface')

    # Sidebar for model selection
    st.sidebar.header('Model Configuration')
    model_path = st.sidebar.file_uploader('Upload BiLSTM Model (.pth)', type=['pth'])

    # Main content area
    st.header('Outfit Variant Scoring')

    # Image upload sections
    st.subheader('Base Outfit Images')
    base_outfit_images = st.file_uploader('Upload Base Outfit Images (Top to Bottom)',
                                          accept_multiple_files=True,
                                          type=['jpg', 'jpeg', 'png'])

    # Variant upload sections
    st.subheader('Outfit Variants')
    num_variants = st.number_input('Number of Variant Sets', min_value=1, max_value=10, value=3)

    variant_sets = []
    for i in range(num_variants):
        st.markdown(f'### Variant Set {i + 1}')
        variant_images = st.file_uploader(f'Upload Variant {i + 1} Images',
                                          accept_multiple_files=True,
                                          type=['jpg', 'jpeg', 'png'],
                                          key=f'variant_{i}')
        variant_sets.append(variant_images)

    # Score button
    if st.button('Calculate Outfit Scores'):
        if model_path and base_outfit_images and all(variant_sets):
            try:
                # Initialize the app
                app = OutfitScoringApp(model_path)
                
                # Calculate scores
                scores = app.score_outfit(base_outfit_images, variant_sets)
                
                cols = st.columns(num_variants) 
                
                for i, image in enumerate(scores):
                    # Dynamically select the column
                    col = cols[i % num_variants]
                    # Display the image in the selected column
                    col.image(image, caption=f"Score {scores[i]}", width=150)

            except Exception as e:
                st.error(f'Error processing outfit: {str(e)}')
        else:
            st.warning('Please upload a model and all required images')


if __name__ == '__main__':
    main()