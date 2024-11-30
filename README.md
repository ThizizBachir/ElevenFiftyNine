# 🌟 ZEN TSYP Challenge Solution
<p align="center">
![logoZen](https://github.com/user-attachments/assets/cd1627b4-41d5-45e5-bd3b-6f3bb4e73a2c)
</p>

---

## 🚀 Introduction

The ZEN TSYP Challenge is an opportunity to innovate and solve real-world problems by utilizing advanced AI and 3D technologies. Our project creates a pipeline to automate and enhance human 3D modeling, including features like body segmentation, skin tone detection, and seamless integration with 3D design tools.

---

## 💡 Solution Overview

Our solution comprises several modules working in harmony:  
- AI-driven body and face segmentation.  
- Skin tone, age, and gender detection for personalization.  
- Anthropometric scaling for accurate 3D representation.  
- Integration with tools like Blender and Marvelous Designer for 3D modeling.  
- Recommendation System: Recommends a 3D model that suits the client based on extracted features.  

---

## 🛠️ Technologies Used

### AI Models and Libraries
1. **Meta Sapiens Model**: For body segmentation. [📄 Research Paper](https://ar5iv.labs.arxiv.org/html/2408.12569)  
2. **DeepFace Library**: For face segmentation. [🔗 GitHub Repository](https://github.com/serengil/deepface)  
3. **Skin Tone Detection Model**: Implements histogram-based or ML-driven skin detection. [📄 Related Paper](https://arxiv.org/pdf/2103.14191.pdf)  
4. **Gender and Age Detection Model**: [📄 Research Paper](https://arxiv.org/abs/1708.08039)  

### Tools and Platforms
1. **HumGen3D**: Anthropometric scaling and integration. [🔗 Official Documentation](https://www.humgen3d.com)  
2. **Docker**: Containerization for deployment. [🔗 Documentation](https://www.docker.com/get-started)  
3. **Azure**: Cloud platform to run Docker containers. [🔗 Azure Container Instances](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-overview)  
4. **Blender**: 3D rendering and modeling with the HumGen3D add-on. [🔗 Add-On Documentation](https://www.humgen3d.com/blender)  
5. **Marvelous Designer/Clo3D**: For designing realistic 3D garments. [🔗 Official Website](https://www.marvelousdesigner.com)  

---

## 🏗️ System Architecture

Below is an overview of the system architecture:  
1. **Data Processing**: Body and face segmentation with AI models.  
2. **Feature Extraction**: Skin tone detection, gender, and age estimation.  
3. **Anthropometric Scaling**: Normalize data to scale (0–1) for use with HumGen3D.  
4. **3D Model Generation**: Design clothing in Marvelous Designer and integrate with Blender for rendering.  

### 🎨 Architecture Diagram
<p align="center">
---![conecept-map](https://github.com/user-attachments/assets/fe936b0c-3f18-44f0-a1ec-92b86394f11b)
![diagram](https://github.com/user-attachments/assets/02cb5ad8-8f0d-4e4e-bc65-0b0950114994)
</p>

## ⚙️ Installation and Setup

### Prerequisites
- Install Docker and Docker Compose.  
- Python 3.8+ and necessary dependencies (requirements.txt).  
- Blender 3.0+ with the HumGen3D add-on installed.  

### Steps
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-repo-name
   cd your-repo-name
2. **Set up the environment**:  
   ```bash
   python -m venv env
   source env/bin/activate   # For Linux/MacOS
   env\Scripts\activate      # For Windows
   pip install -r requirements.txt
3. **Download the Sapiens Segmentation Model**:  
   ```bash
   mkdir model
   cd model
   wget https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens-seg-1b.pt

