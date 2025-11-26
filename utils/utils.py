import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def create_efficientnet_model(num_class: int = 5):
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    
    num_features = model.classifier[-1].in_features
    model.classifier[1] = nn.Linear(num_features, num_class)
    model.eval()
    return model

def load_model(path: str, num_class: int = 5):
    model = create_efficientnet_model(num_class)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    try:
        dummy_data = torch.rand(size=(1, 3, 224, 224))
        model = torch.jit.trace(model, dummy_data)
        print('Model converted to TorchScript.')
    except Exception as e:
        print(f"TorchScript conversion failed: {e}")
    return model, checkpoint

CLASS_NAMES = ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic']
MODEL_PATH = Path(__file__).parent.parent / "models" / "model_pretrained_True.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_model = None
_checkpoint = None

def get_model():
    global _model, _checkpoint
    
    if _model is None:
        _model, _checkpoint = load_model(MODEL_PATH)
    
    return _model, _checkpoint

def predict_disease(image):
    try:
        model, _ = get_model()
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1).numpy()[0]
        results = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        return results
    except Exception as e:
        return {"error": f"Prediction failed {str(e)}"}

